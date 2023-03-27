classdef MistralRLEnvClass < rl.env.MATLABEnvironment
    properties
        state;
        Figure = false;
        numUsers = 2;
        numUavs = 1;
        uavPlot;
        userPlot;
        connectionPlot;
        plotFlag = false;
        KEDFlag = true;

        


    end

    methods              
        function this = MistralRLEnvClass()
            numUsers = 2;
            actionInfo = rlFiniteSetSpec([1 2 3 4 5 6 7 8 9]);
            uavCoords = rlNumericSpec([1 3]); 
            userDistance = rlNumericSpec([1 numUsers]);
            userAngleHeading =  rlNumericSpec([1 numUsers]);
            userAnglePitch =    rlNumericSpec([1 numUsers]);
            userD1Distance =    rlNumericSpec([1 numUsers]);
            obsInfo(1) = uavCoords;
            obsInfo(2) = userDistance;
            obsInfo(3) = userAngleHeading;
            obsInfo(4) = userAnglePitch;
            this = this@rl.env.MATLABEnvironment(obsInfo,actionInfo);
        end
        









































        function [NextObs,Reward,IsDone,LoggedSignals] = step(this,Action)
            clc;
            
        %% Constant Declaration ( can replace this function with a function handle later to pass these constants to function)
            randomMovementFlag = true;

            mapResolution = 400;
            moveChance = 70;
            
            uavHeightRel = 300;
            userHeightRel = 1.5;
            freqHz = 160e6;
            moveMatrix = [
                [mapResolution 0]
                [mapResolution mapResolution]
                [0 mapResolution]
                [-mapResolution mapResolution]
                [-mapResolution 0]
                [-mapResolution -mapResolution]
                [0 -mapResolution]
                [mapResolution -mapResolution]
                [0 0]
            ];

            LoggedSignals = this.state;
            %display episode progress
            disp(LoggedSignals.stepCounter{1} / 5+"%");
            LoggedSignals.stepCounter{1} = LoggedSignals.stepCounter{1}+1;
        
        %% Get values of previous state.
        
            disp(LoggedSignals.stepCounter{1} / 5+"%");
            LoggedSignals.stepCounter{1} = LoggedSignals.stepCounter{1}+1;
            previousBaseCoords =        LoggedSignals.baseCoords{1};
            previousUavCoords =         LoggedSignals.uavCoords{1};
            previousUserCoords =        LoggedSignals.userCoords{1};
            previousUserDistance =      LoggedSignals.userDistance{1};
            previousUserAngleHeading =  LoggedSignals.userAngleHeading{1};
            previousUserAnglePitch =    LoggedSignals.userAnglePitch{1};
            previousUserD1Distance =    LoggedSignals.userD1Distance{1};
            previousElevMap =           LoggedSignals.elevMap{1};
            previousForestCoverOneHot = LoggedSignals.forestCoverOneHot{1};
            previousXVector =           LoggedSignals.xVector{1};
            previousYVector =           LoggedSignals.yVector{1};
        
        
        %% move environment variables to next iteration (known as a step).
            %if no action, try cut down on calculations.
            %Action Processing.
            Action = moveMatrix(Action,:);
            Action = [Action(1) Action(2) 0];
        
            % Update Uav Coordinates
            nextUavCoords = previousUavCoords + Action;
        
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
           
            %Update elev map - Wont change will remove later.
            nextElevMap = previousElevMap;
        
            % Update forest cover - Wont change will remove later.
            nextForestCoverOneHot = previousForestCoverOneHot;
            
            % Update base Coords - Wont change will remove later.
            nextBaseCoords = previousBaseCoords;
        
            nextUserCoords = previousUserCoords;
        
            if randomMovementFlag
                moveRNG = randi([0,100],1,1);
                if moveRNG <= moveChance
                    randomMoveset = randi([-1,1],this.numUsers,2)*mapResolution;
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
                        elevMapX = nextUserCoords(userCtr,1)/mapResolution;
                        elevMapY = nextUserCoords(userCtr,2)/mapResolution;
                        userElev = nextElevMap(elevMapX, elevMapY);
                        userElev = userElev+userHeightRel;
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
        
           
        
            %also include base as receiver.
            %% LOS vect seems to give (possible) false positives for LOS, users seem to be always visible.
            [LOS_vect,~,~, ~, nextUserD1Distance,~,~] = get_LOS_vect(nextElevMap, nextForestCoverOneHot, nextUavCoords(:,1), nextUavCoords(:,2), nextUserCoords(:,1),nextUserCoords(:,2), ...
                uavHeightRel,userHeightRel,mapResolution,previousXVector,previousYVector,freqHz,this.KEDFlag );
        
            
            disp(LOS_vect)
            disp("Distance from yawPitchDistance:")
            disp(nextUserDistance)
            disp("d1_dist_vect")
            disp(nextUserD1Distance)
        
            %nextUserD1Distance = reshape(nextUserD1Distance,1,this.numUsers);
        
            % vector containing the pathloss metrics for all user connections, dont
            % forget to add base pathloss.
            pathlossVector = pathloss_model(nextUserDistance, freqHz, LOS_vect);
            %% add base pathloss
        
            % add LOS for base.
        
            %[baseDistance,~,~] = yawPitchDistanceFromCoords(nextUavCoords, nextBaseCoords);
            %pathlossVector(5) = pathloss_model(baseDistance,freqHz);
            
            Reward = min(pathlossVector); 
            
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
            LoggedSignals.elevMap{1} =          nextElevMap;
            LoggedSignals.forestCoverOneHot{1} =nextForestCoverOneHot;
        
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
            
            if this.plotFlag
                updatePlot(this);
            end
        end
        

































        % Reset environment to initial state and output initial observation
        function [InitialObservation, LoggedSignal] = reset(this)
            map_resolution = 400;
            uavHeightRel = 300;
            userHeightRel = 1.5;
            freqHz = 160e6;
        
            randUserPositionFlag = true;
            randUavPositionFlag = true;
        
        
            %Actual elevMap and forest cover - taken from orignal mistralRl script.
            [forestCover_onehot, elevMap, xVector, yVector ] = generateTerrain(map_resolution);
        
            %actual base coords from simulator
            baseCoords = [13200 13200 0];
        
            %random uav postion each step.
            if randUavPositionFlag
                uavCoords = randi([0 71],1,2);
                uavCoords = [uavCoords(1)*map_resolution uavCoords(2)*map_resolution uavHeightRel ];
            else
                uavCoords = [4000,4000,uavHeightRel];
            end 
            
            if randUserPositionFlag
                userCoordsInd = randi([1,70], this.numUsers,2);
                userCoords = userCoordsInd * map_resolution;
                userCoords(:,3) = 0;
                for userCtr = 1:this.numUsers
                    userIndexX = userCoordsInd(userCtr,1);
                    userIndexY = userCoordsInd(userCtr,2);
                    userCoords(userCtr,3) = elevMap(userIndexX,userIndexY);
                end
                userCoords(:,3) = userCoords(:,3)+userHeightRel;
            else
                %initalise all users in same position.
                userCoords = zeros(this.numUsers,3);
                userCoords(:,1) = 24000;
                userCoords(:,2) = 24000;
                userCoords(:,3) = elevMap(61,61);
                userCoords(:,3) = userCoords(:,3)+userHeightRel;
            end
        
            userDistance = zeros(1,this.numUsers);
            userAngleHeading = zeros(1,this.numUsers);
            userAnglePitch = zeros(1,this.numUsers);
        
            for userCtr = 1:this.numUsers
                [userDistance(1,userCtr), userAngleHeading(1,userCtr), userAnglePitch(1,userCtr)] = yawPitchDistanceFromCoords(uavCoords, userCoords(userCtr,:));
            end
            userD1Distance = zeros(1,this.numUsers);
        
            LoggedSignal.stepCounter{1} = 1;
            LoggedSignal.baseCoords{1} = baseCoords;
            LoggedSignal.uavCoords{1} = uavCoords;
            LoggedSignal.userCoords{1} = userCoords;
            LoggedSignal.userDistance{1} = userDistance;
            LoggedSignal.userAngleHeading{1} = userAngleHeading;
            LoggedSignal.userAnglePitch{1} = userAnglePitch;
            LoggedSignal.userD1Distance{1} = userD1Distance;
            LoggedSignal.elevMap{1} = elevMap;
            LoggedSignal.forestCoverOneHot{1} = forestCover_onehot;
            LoggedSignal.xVector{1} = xVector;
            LoggedSignal.yVector{1} = yVector;

            LoggedSignal.State{1} = uavCoords;
            LoggedSignal.State{2} = userDistance;
            LoggedSignal.State{3} = userAngleHeading;
            LoggedSignal.State{4} = userAnglePitch;

            InitialObservation = LoggedSignal.State;
            this.state = LoggedSignal;
            %notifyEnvUpdated(this);
            disp("plot called for reset");
            %plot(this,true);


            P_Rx_map = zeros(size(this.state.elevMap{1},1),size(this.state.elevMap{1},2),this.numUavs);
            this.state.max_P_Rx_map{1} = max(P_Rx_map,[],3);

            disp(this.state.uavCoords{1})
            if this.plotFlag
                plot(this);
            end
            
        end


        function plot(this)
            %clf(this.figure, "reset");
            if this.Figure == false
                this.Figure = figure("Name","Training Visualisation.","NumberTitle","off","Color","[1 1 1]");
            end
            clf;
            ax1 = axes;
            contour(this.state.xVector{1},this.state.yVector{1},this.state.elevMap{1}',30);
            axis equal
            view(2)
            ax2 = axes;
            P_Rx_map_plot = surf(this.state.xVector{1},this.state.yVector{1},this.state.max_P_Rx_map{1}');
            
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
            axis([min(this.state.xVector{1}) max(this.state.xVector{1}) min(this.state.yVector{1}) max(this.state.yVector{1})]);
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
            

            %this.connectionPlot = plot([duplicateUavCoordsX, this.state.userCoords{1}(:,1)], [duplicateUavCoordsY, this.state.userCoords{1}(:,2)],"LineWidth",1,"color","black");
            %this.connectionPlot = plot([this.state.uavCoords{1}(:,1), this.state.userCoords{1}(:,1)], [this.state.uavCoords{1}(:,2), this.state.userCoords{1}(:,2)],"LineWidth",1,"color","black");
            
        end

        function updatePlot(this)
            assignin('base','uavXCoords',this.state.uavCoords{1}(:,1))
            assignin('base','uavYCoords',this.state.uavCoords{1}(:,2))
            assignin('base','userXCoords',this.state.userCoords{1}(:,1))
            assignin('base','userYCoords',this.state.userCoords{1}(:,2))
            refreshdata(this.uavPlot)
            refreshdata(this.userPlot)
            delete(this.connectionPlot)

            %number of connection must adapt to include base connections
            %and multiple uavs connections and there allocated users.
            numConnections = this.numUsers * this.numUavs;

            xPointsForPlot = zeros(numConnections*2,1);
            yPointsForPlot = zeros(numConnections*2,1);
            xPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,1);
            yPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,2);
            xPointsForPlot(2:2:end) = this.state.userCoords{1}(:,1);
            yPointsForPlot(2:2:end) = this.state.userCoords{1}(:,2);

            this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");

            
            
            %this.connectionPlot = plot([duplicateUavCoordsX, this.state.userCoords{1}(:,1)], [duplicateUavCoordsY, this.state.userCoords{1}(:,2)],"LineWidth",1,"color","black");
            %this.connectionPlot = plot([this.state.uavCoords{1}(:,1), this.state.userCoords{1}(:,1)], [this.state.uavCoords{1}(:,2), this.state.userCoords{1}(:,2)],"LineWidth",1,"color","black");
            drawnow;
        end
       

    end
end
