classdef MistralRLEnvClass < rl.env.MATLABEnvironment
    properties


        %% state persists many of the dynamic environmental variables
        state;

        %plotting variables
        plotFlag = true;
        Figure = false;
        uavPlot;
        userPlot;
        connectionPlot;

        numUsers = 2;
        numUavs = 1;
        
        %% environmental Constants.
        uavHeightRel = 1000;
        userHeightRel = 1.5;
        KEDFlag = true;
        freqHz = 160e6;
            %mapResolution is hardcoded for now will change soon.
        mapResolution = 400;
        forestCoverOneHot;
        elevMap;
        xVector;
        yVector;

        %% user and uav behavioural flags.
        randomUserMovementFlag = true;
        randomUserMovementChance = 70;
           % random postions for uavs and users at the start of each
           % episode
        randomUserPositionFlag = true;
        randomUavPositionFlag = true;
    end

    methods  

        function this = MistralRLEnvClass()
            %throws error if number of users / uavs arent hardcoded before
            %being passed to super constructor.
            numUsers = 2;

            %setting out our 9 actions.
            actionInfo = rlFiniteSetSpec([1 2 3 4 5 6 7 8 9]);

            %our observations of the state, these data structures contain
            %data used to train our models.
            uavCoords = rlNumericSpec([1 3]); 
            userDistance = rlNumericSpec([1 numUsers]);

                %Heading angle is essentially the North - South orientation
                %of user from the drone in radians.
            userAngleHeading =  rlNumericSpec([1 numUsers]);

                %pitch angle is just the angle between the line of sight to
                %the user and a line straight to the ground from the uav.
            userAnglePitch =    rlNumericSpec([1 numUsers]);

                %d1 not currently being used until I verify if i am using
                %the LOS_vect function correctly.
            userD1Distance =    rlNumericSpec([1 numUsers]);

            obsInfo(1) = uavCoords;
            obsInfo(2) = userDistance;
            obsInfo(3) = userAngleHeading;
            obsInfo(4) = userAnglePitch;
            %obsInfo(5) = userD1Distance;

            %create RL envvironment by calling superclass constructor.
            this = this@rl.env.MATLABEnvironment(obsInfo,actionInfo);
        end

        %In the case of a Deep Q Network, Models train using the below,
        %step and reset function like so.
        %   1. Step() function, i.e. make a move.
        %   2. Keep making moves until it is time to retrain the network.
        %   e.g 50 moves.
        %   3. Use experience to retrain network on batch of experience.
        %   4. At the end of episode (e.g 500 - 2000).
        %   5. Reset() function, reset environment to initial (or new
        %   random state).
        %   6. Continue to next episode, retain network and experiences.


        % Reset environment to initial state and output initial observation
        %generally after 500 - 2000 moves, which we refer to as steps.
        function [InitialObservation, LoggedSignal] = reset(this)

            %Actual elevMap and forest cover - taken from orignal mistralRl script.
            [this.forestCoverOneHot, this.elevMap, this.xVector, this.yVector ] = generateTerrain(this.mapResolution);
        
            %actual base coords from simulator
            baseCoords = [13200 13200 0];
        
            %random uav postion each step.
            if this.randomUavPositionFlag
                uavCoords = randi([0 71],1,2);
                uavCoords = [uavCoords(1)*this.mapResolution uavCoords(2)*this.mapResolution this.uavHeightRel ];
            else
                %hardcoded, consistent uav start point.
                uavCoords = [4000,4000,this.uavHeightRel];
            end 
            
            %I will later integrate the UserMobility class but for now
            %users move randomly with a set chance to move each time the
            %uav does. 

            %later we can also store and increment a sample time across multiple steps,
            %so we can move users at appropriate speeds.
            if this.randomUserPositionFlag
                userCoordsInd = randi([1,70], this.numUsers,2);
                userCoords = userCoordsInd * this.mapResolution;
                userCoords(:,3) = 0;
                for userCtr = 1:this.numUsers
                    userIndexX = userCoordsInd(userCtr,1);
                    userIndexY = userCoordsInd(userCtr,2);
                    userCoords(userCtr,3) = this.elevMap(userIndexX,userIndexY);
                end
                userCoords(:,3) = userCoords(:,3)+this.userHeightRel;
            else
                %initalise all users in same position, purely for testing
                % exploration policies - ignore.
                userCoords = zeros(this.numUsers,3);
                userCoords(:,1) = 24000;
                userCoords(:,2) = 24000;
                userCoords(:,3) = this.elevMap(61,61);
                userCoords(:,3) = userCoords(:,3)+this.userHeightRel;
            end
            
            %I had my own distance formula written I integrated your
            %pathloss calcualtions, but my calcs seem to give exact same
            %values.
            userDistance = zeros(1,this.numUsers);
            userAngleHeading = zeros(1,this.numUsers);
            userAnglePitch = zeros(1,this.numUsers);
        
            for userCtr = 1:this.numUsers
                [userDistance(1,userCtr), userAngleHeading(1,userCtr), userAnglePitch(1,userCtr)] = yawPitchDistanceFromCoords(uavCoords, userCoords(userCtr,:));
            end
            userD1Distance = zeros(1,this.numUsers);
            
            %I had to reformat the step and reset fucntions to a subclass
            %of RLEnvironment to facilitate the visualisation of training,
            %this also means this LoggedSignal data structure might not be
            %needed anymore, will remove later if so for simplicity.
            LoggedSignal.stepCounter{1} = 1;
            LoggedSignal.baseCoords{1} = baseCoords;
            LoggedSignal.uavCoords{1} = uavCoords;
            LoggedSignal.userCoords{1} = userCoords;
            LoggedSignal.userDistance{1} = userDistance;
            LoggedSignal.userAngleHeading{1} = userAngleHeading;
            LoggedSignal.userAnglePitch{1} = userAnglePitch;
            LoggedSignal.userD1Distance{1} = userD1Distance;

            LoggedSignal.State{1} = uavCoords;
            LoggedSignal.State{2} = userDistance;
            LoggedSignal.State{3} = userAngleHeading;
            LoggedSignal.State{4} = userAnglePitch;

            InitialObservation = LoggedSignal.State;
            this.state = LoggedSignal;
            %for plotting - taken from original mistral script.
            P_Rx_map = zeros(size(this.elevMap,1),size(this.elevMap,2),this.numUavs);
            this.state.max_P_Rx_map{1} = max(P_Rx_map,[],3);
            
            %plot initial environment.
            if this.plotFlag
                plot(this);
            end
            
        end
        
        function [NextObs,Reward,IsDone,LoggedSignals] = step(this,Action)
            clc;
            %dont worry about this matrix we simply use it increment the
            %uav Position.
            moveMatrix = [
                [1 0]
                [1 1]
                [0 1]
                [-1 1]
                [-1 0]
                [-1 -1]
                [0 -1]
                [1 -1]
                [0 0]
            ] * this.mapResolution;

        %% Get values of previous state.
            LoggedSignals = this.state;
            disp("Step Counter")
            disp(LoggedSignals.stepCounter{1});
            LoggedSignals.stepCounter{1} = LoggedSignals.stepCounter{1}+1;
            
            previousBaseCoords =        LoggedSignals.baseCoords{1};
            previousUavCoords =         LoggedSignals.uavCoords{1};
            previousUserCoords =        LoggedSignals.userCoords{1};
        
        
        %% move environment variables to next iteration (known as a step).
            %if no action, try cut down on calculations.
            %Action Processing.
            Action = moveMatrix(Action,:);
            Action = [Action(1) Action(2) 0];
        
            % Update Uav Coordinates
            nextUavCoords = previousUavCoords + Action;
            
            %enforce border on movement.
            if nextUavCoords(1) > 28400
                nextUavCoords(1) = 28400;
            elseif nextUavCoords(1) <  0
                nextUavCoords(1) = 0;
            end 
        
            if nextUavCoords(2) > 28000
                nextUavCoords(2) = 28000;
            elseif nextUavCoords(2) <  0
                nextUavCoords(2) = 0;
            end 
           
            % Update base Coords - keeping consistent with loggedsignals
            % they wont change so will remove this later.
            nextBaseCoords = previousBaseCoords;
        
            nextUserCoords = previousUserCoords;
            
            %move users according to (Chance to move -> random movement)
            if this.randomUserMovementFlag
                moveRNG = randi([0,100],1,1);
                if moveRNG <= this.randomUserMovementChance
                    randomMoveset = randi([-1,1],this.numUsers,2)*this.mapResolution;
                    randomMoveset(:,3) = 0;
                    nextUserCoords = nextUserCoords + randomMoveset;
                    for userCtr = 1:this.numUsers
                        if nextUserCoords(userCtr,1) > 28400
                            nextUserCoords(userCtr,1) = 28400;
                        elseif nextUserCoords(userCtr,1) <  400
                            nextUserCoords(userCtr,1) = 400;
                        end 
                    
                        if nextUserCoords(userCtr,2) > 28000
                            nextUserCoords(userCtr,2) = 28000;
                        elseif nextUserCoords(userCtr,2) <  400
                            nextUserCoords(userCtr,2) = 400;
                        end
                        elevMapX = nextUserCoords(userCtr,1)/this.mapResolution;
                        elevMapY = nextUserCoords(userCtr,2)/this.mapResolution;
                        userElev = this.elevMap(elevMapX, elevMapY);
                        userElev = userElev+this.userHeightRel;
                        nextUserCoords(userCtr,3) = userElev;
                    end
                end
            end
        
        
            %User Distance
            %User Angle A (Birds eye view angle - known as heading)
            %User Angle B (Angle to Ground - known as pitch)
            nextUserDistance = zeros(1,this.numUsers);
            nextUserAngleHeading = zeros(1,this.numUsers);
            nextUserAnglePitch = zeros(1,this.numUsers);
        
            for userCtr = 1:this.numUsers
                [nextUserDistance(1,userCtr), nextUserAngleHeading(1,userCtr), nextUserAnglePitch(1,userCtr)] = yawPitchDistanceFromCoords(nextUavCoords, nextUserCoords(userCtr,:));
            end
        
           
        
            %Uav to base connections work fine within the model but I am
            %going to remove for now until models are training
            %appropriately.

            %% LOS vect seems to give (possible) false positives for LOS, users seem to be always visible.
            [LOS_vect,~,~, ~, nextUserD1Distance,~,~] = get_LOS_vect(this.elevMap, this.forestCoverOneHot, nextUavCoords(:,1), nextUavCoords(:,2), nextUserCoords(:,1),nextUserCoords(:,2), ...
                this.uavHeightRel,this.userHeightRel,this.mapResolution,this.xVector,this.yVector,this.freqHz,this.KEDFlag );
        
            disp("Line of sight boolean from 'LOS_VECT.m'")
            disp(LOS_vect)
            disp("Distance from custom function 'yawPitchDistanceFromCoords.m'")
            disp(nextUserDistance)
            disp("d1_distances from 'LOS_Vect.m'")
            disp(nextUserD1Distance)
        
            %nextUserD1Distance = reshape(nextUserD1Distance,1,this.numUsers);
        
            % vector containing the pathloss metrics for all user
            % connections, I must 
            pathlossVector = pathloss_model(nextUserDistance, this.freqHz, LOS_vect);
        
            %[baseDistance,~,~] = yawPitchDistanceFromCoords(nextUavCoords, nextBaseCoords);
            %pathlossVector(end+1) = pathloss_model(baseDistance,this.freqHz);
            
            %weighted sum for rewards maybe, worst pathloss - mean pathloss
            %also.


            %% Current reward is signal with most pathloss, but this may be causing problems in training with multiple connection scenarios
            Reward = min(pathlossVector);
            %% How about a weighted sum e.g ( best pathloss gets weight of 1, 2nd best gets 1.25, worst gets 1.5)
            

            %IsDone flag will stop an epsisode and activate the reset fucntion,
            %episodes will reset on their own at a certain amount of steps, but
            %this flag could be triggered given a certain condition, eg. uav runs
            %out of fuel.
            IsDone = false;
        
        
        %% Return variables for next step.
        
            LoggedSignals.baseCoords{1} =       nextBaseCoords;
            LoggedSignals.uavCoords{1} =        nextUavCoords;
            LoggedSignals.userCoords{1} =       nextUserCoords;
            LoggedSignals.userDistance{1} =     nextUserDistance;
            LoggedSignals.userAngleHeading{1} = nextUserAngleHeading;
            LoggedSignals.userAnglePitch{1} =   nextUserAnglePitch;
            LoggedSignals.userD1Distance{1} =   nextUserD1Distance;
        
            LoggedSignals.State{1} = nextUavCoords;
            LoggedSignals.State{2} = nextUserDistance;
            LoggedSignals.State{3} = nextUserAngleHeading;
            LoggedSignals.State{4} = nextUserAnglePitch;
            %LoggedSignals.State{} = nextUserD1Distance;
            %LoggedSignals.State{} = nextBaseCoords;
        
            NextObs = LoggedSignals.State;
            this.state = LoggedSignals;
            
            disp("Reward: "+Reward)
            disp("Uav Position:")
            disp("    "+nextUavCoords(1)+"    "+nextUavCoords(2)+"    "+nextUavCoords(3))
            disp("User Postions:")
            disp(nextUserCoords)
            disp(newline)
            
            %update Plot.
            if this.plotFlag
                updatePlot(this);
            end
        end
        
        function plot(this)
            %clf(this.figure, "reset");
            if this.Figure == false
                this.Figure = figure("Name","Training Visualisation.","NumberTitle","off","Color","[1 1 1]");
            end
            clf;
            ax1 = axes;
            contour(this.xVector,this.yVector,this.elevMap',30);
            axis equal
            view(2)
            ax2 = axes;
            P_Rx_map_plot = surf(this.xVector,this.yVector,this.state.max_P_Rx_map{1}');
            
            view(2);
            P_Rx_map_plot.CDataSource = ['max_P_Rx_map' char(39)];
            caxis([-110 -50]);
            axis equal
            alpha(0.01);
            shading flat
            linkaxes([ax1,ax2])
            ax2.Visible = 'off';
            ax2.XTick = [];
            ax2.YTick = [];
            axis([min(this.xVector) max(this.xVector) min(this.yVector) max(this.yVector)]);
            colorbar
            set([ax1,ax2],'Position',[.17 .11 .685 .815]);
            hold on

            this.uavPlot = plot(this.state.uavCoords{1}(:,1), this.state.uavCoords{1}(:,2), '^w', 'MarkerFaceColor','k', 'MarkerSize', 10);
            this.uavPlot.XDataSource = "uavXCoords";
            this.uavPlot.YDataSource = "uavYCoords";

            this.userPlot = plot(this.state.userCoords{1}(:,1), this.state.userCoords{1}(:,2), 'ow', 'MarkerFaceColor','k', 'MarkerSize', 10);
            this.userPlot.XDataSource = "userXCoords";
            this.userPlot.YDataSource = "userYCoords";

            numConnections = this.numUsers * this.numUavs;
            
            xPointsForPlot = zeros(numConnections*2,1);
            yPointsForPlot = zeros(numConnections*2,1);
            xPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,1);
            yPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,2);
            xPointsForPlot(2:2:end) = this.state.userCoords{1}(:,1);
            yPointsForPlot(2:2:end) = this.state.userCoords{1}(:,2);

            this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");
            
        end

        function updatePlot(this)
            %variables must be moved to the base workspace to be used as
            %the data source for plots.
            assignin('base','uavXCoords',this.state.uavCoords{1}(:,1))
            assignin('base','uavYCoords',this.state.uavCoords{1}(:,2))
            assignin('base','userXCoords',this.state.userCoords{1}(:,1))
            assignin('base','userYCoords',this.state.userCoords{1}(:,2))
            refreshdata(this.uavPlot)
            refreshdata(this.userPlot)
            delete(this.connectionPlot)

            %number of connection must adapt to include base connections
            %and multiple uavs connections and there allocated users.
            %will change later to work kmeans allocation of users to uavs.
            numConnections = this.numUsers * this.numUavs;

            

            xPointsForPlot = zeros(numConnections*2,1);
            yPointsForPlot = zeros(numConnections*2,1);
            xPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,1);
            yPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,2);
            xPointsForPlot(2:2:end) = this.state.userCoords{1}(:,1);
            yPointsForPlot(2:2:end) = this.state.userCoords{1}(:,2);

            this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");

            drawnow;
        end
       

    end
end
