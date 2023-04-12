classdef MistralRLEnvClass < rl.env.MATLABEnvironment
    properties

        %% Notes
        % ~ Move Flags to env constructor and set them after calling
        %   super constructor?
        % ~ Integrate UserMobility class.
        % ~ Integrate dynamic map resolution.
        % ~ Calculate d1 dist in reset function.


        state;

        plotFlag;
        displayStepFlag = true;
        Figure = false;
        uavPlot;
        userPlot;
        centroidPlot;
        connectionPlot;

        numUsers;
        numUavs = 1;
        %greatestInd
        
        uavHeightRel = 1000;
        userHeightRel = 1.5;
        baseHeightRel = 20;
        KEDFlag = true;
        freqHz = 160e6;
        rewardGivenForLogging;

        mapResolution = 400;
        forestCoverOneHot;
        elevMap;
        xVector;
        yVector;

        centroid;
        radiusPlot;

        weightedSumReward = false;
        pathlossWeights = [1.5,1];
        standardGreatestPathlossReward = true;
        centroidMeanNormalisedReward = false;
        randomUserMovementFlag = true;
        randomUserMovementChance = 70;
        pathlossVector;

        randomUserPositionFlag = true;
        randomUavPositionFlag = true;
    end

    




    methods  
%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function this = MistralRLEnvClass(plotFlag,numUsers)
            actionInfo = rlFiniteSetSpec([1 2 3 4 5 6 7 8]);

            uavCoords = rlNumericSpec([1 3]); 
            userDistance = rlNumericSpec([1 numUsers]);
            userAngleHeading =  rlNumericSpec([1 numUsers]);
            userAnglePitch =    rlNumericSpec([1 numUsers]);
            userD1Distance =    rlNumericSpec([1 numUsers]);

            %{ oringinal methods 1 * numUsers
            %obsInfo(1) = userDistance;
            %obsInfo(2) = userAngleHeading;
            %obsInfo(3) = userAnglePitch;
            %}

            %{
             %2 users input in seperate layers + pathloss also
            obsInfo(1) = rlNumericSpec([1 1]);
            obsInfo(2) = rlNumericSpec([1 1]);
            obsInfo(3) = rlNumericSpec([1 1]);
            obsInfo(4) = rlNumericSpec([1 1]);
            obsInfo(5) = rlNumericSpec([1 1]);
            obsInfo(6) = rlNumericSpec([1 1]);
            obsInfo(7) = rlNumericSpec([1 1]);
            obsInfo(8) = rlNumericSpec([1 1]);
            %}

            obsInfo(1) = rlNumericSpec([1 1]);
            obsInfo(2) = rlNumericSpec([1 1]);
            obsInfo(3) = rlNumericSpec([1 1]);


            this = this@rl.env.MATLABEnvironment(obsInfo,actionInfo);


            this.plotFlag = plotFlag;
            this.numUsers = numUsers;

            validateEnvironment(this)
        end







%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function [InitialObservation, LoggedSignal] = reset(this)
            [this.forestCoverOneHot, this.elevMap, this.xVector, this.yVector ] = generateTerrain(this.mapResolution);
            baseCoords = [13200 13200 0];

            if this.randomUavPositionFlag
                uavCoords = randi([0 71],1,2);
                uavCoords = [uavCoords(1)*this.mapResolution uavCoords(2)*this.mapResolution this.uavHeightRel ];
            else
                uavCoords = [4000,4000,this.uavHeightRel];
            end 

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
                userCoords = zeros(this.numUsers,3);
                userCoords(:,1) = 24000;
                userCoords(:,2) = 24000;
                userCoords(:,3) = this.elevMap(61,61);
                userCoords(:,3) = userCoords(:,3)+this.userHeightRel;
            end
            

            userDistance = zeros(1,this.numUsers);
            userAngleHeading = zeros(1,this.numUsers);
            userAnglePitch = zeros(1,this.numUsers);
        
            for userCtr = 1:this.numUsers
                [userDistance(1,userCtr), userAngleHeading(1,userCtr), userAnglePitch(1,userCtr)] = yawPitchDistanceFromCoords(uavCoords, userCoords(userCtr,:));
            end
            userD1Distance = zeros(1,this.numUsers);
            
            LoggedSignal.stepCounter{1} = 0;
            LoggedSignal.baseCoords{1} = baseCoords;
            LoggedSignal.uavCoords{1} = uavCoords;
            LoggedSignal.userCoords{1} = userCoords;
            LoggedSignal.userDistance{1} = userDistance;
            LoggedSignal.userAngleHeading{1} = userAngleHeading;
            LoggedSignal.userAnglePitch{1} = userAnglePitch;
            LoggedSignal.userD1Distance{1} = userD1Distance;

            %InitialObservation{1} = userDistance;
            %InitialObservation{2} = userAngleHeading;
            %InitialObservation{3} = userAnglePitch;
            %{
            InitialObservation{1} = userDistance(1,1);
            InitialObservation{2} = userAngleHeading(1,1);
            InitialObservation{3} = userAnglePitch(1,1);
            InitialObservation{4} = -90;

            InitialObservation{5} = userDistance(1,2);
            InitialObservation{6} = userAngleHeading(1,2);
            InitialObservation{7} = userAnglePitch(1,2);
            InitialObservation{8} = -90;
            %}


            this.centroid = [mean(userCoords(:,1)), mean(userCoords(:,2)), this.uavHeightRel ];

            [~,grInd] = max(userDistance);
            this.centroid = userCoords(grInd,:);
            [centroidDistance, centroidAngleHeading, centroidAnglePitch] = yawPitchDistanceFromCoords(uavCoords,this.centroid);


            InitialObservation{1} = centroidDistance;
            InitialObservation{2} = centroidAngleHeading;
            InitialObservation{3} = centroidAnglePitch;



            this.state = LoggedSignal;

            %for plotting - taken from original mistral script.
            P_Rx_map = zeros(size(this.elevMap,1),size(this.elevMap,2),this.numUavs);
            this.state.max_P_Rx_map{1} = max(P_Rx_map,[],3);
            
            %plot initial environment.
            if this.plotFlag
                plot(this);
            end
        end






%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function [NextObs,Reward,IsDone,LoggedSignals] = step(this,Action)
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

            LoggedSignals = this.state;
            LoggedSignals.stepCounter{1} = LoggedSignals.stepCounter{1}+1;
            
            previousUavCoords =         LoggedSignals.uavCoords{1};
            previousUserCoords =        LoggedSignals.userCoords{1};

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
        
            nextUserDistance = zeros(1,this.numUsers);
            nextUserAngleHeading = zeros(1,this.numUsers);
            nextUserAnglePitch = zeros(1,this.numUsers);
        
            for userCtr = 1:this.numUsers
                [nextUserDistance(1,userCtr), nextUserAngleHeading(1,userCtr), nextUserAnglePitch(1,userCtr)] = yawPitchDistanceFromCoords(nextUavCoords, nextUserCoords(userCtr,:));
            end

            %% LOS vect seems to give (possible) false positives for LOS, users seem to be always visible.
            [LOS_vect,~,~, PL_diff_vect, nextUserD1Distance,~,~] = get_LOS_vect(this.elevMap, this.forestCoverOneHot, nextUavCoords(:,1), nextUavCoords(:,2), nextUserCoords(:,1),nextUserCoords(:,2), ...
                this.uavHeightRel,this.userHeightRel,this.mapResolution,this.xVector,this.yVector,this.freqHz, true );
            disp(LOS_vect)
            
       
        
            %nextUserD1Distance = reshape(nextUserD1Distance,1,this.numUsers);
        
            this.pathlossVector = pathloss_model(nextUserDistance, this.freqHz, LOS_vect);

            %works for 1 user no 1 connection
            %this.pathlossVector = this.pathlossVector - pathloss_model(this.uavHeightRel,this.freqHz);
        
            LoggedSignals.uavCoords{1} =        nextUavCoords;
            LoggedSignals.userCoords{1} =       nextUserCoords;
            LoggedSignals.userDistance{1} =     nextUserDistance;
            LoggedSignals.userAngleHeading{1} = nextUserAngleHeading;
            LoggedSignals.userAnglePitch{1} =   nextUserAnglePitch;
            LoggedSignals.userD1Distance{1} =   nextUserD1Distance;
        
            %NextObs{1} = nextUserDistance;
            %NextObs{2} = nextUserAngleHeading;
            %NextObs{3} = nextUserAnglePitch;
            
            %{
            NextObs{1} = nextUserDistance(1,1);
            NextObs{2} = nextUserAngleHeading(1,1);
            NextObs{3} = nextUserAnglePitch(1,1);
            NextObs{4} = this.pathlossVector(1);

            NextObs{5} = nextUserDistance(1,2);
            NextObs{6} = nextUserAngleHeading(1,2);
            NextObs{7} = nextUserAnglePitch(1,2);
            NextObs{8} = this.pathlossVector(2);
            %}
            
            this.centroid = [mean(nextUserCoords(:,1)), mean(nextUserCoords(:,2)), this.uavHeightRel ];
            

            [~,grInd] = max(nextUserDistance);
            this.centroid = nextUserCoords(grInd,:);
            this.centroid(1,3) = 1000;
            [centroidDistance, centroidAngleHeading, centroidAnglePitch] = yawPitchDistanceFromCoords(nextUavCoords,this.centroid);
            

            
            NextObs{1} = centroidDistance;
            NextObs{2} = centroidAngleHeading;
            NextObs{3} = centroidAnglePitch;

            this.state = LoggedSignals;

            %% only calculate reward after this point.
            
            %{
            if this.weightedSumReward 
                Reward = weightedSumRewardFunction(this);
            elseif this.standardGreatestPathlossReward 
                Reward = standardRewardFunction(this);
            elseif this.centroidMeanNormalisedReward
                Reward = centroidMeanNormalisedRewardFunction(this);
            end
            %}

            %change reward to dist from previous greatestInd.
            %reward = -nextUserdistance(1,this.greatestInd);
            Reward = -(centroidDistance);
            this.rewardGivenForLogging = Reward;

            if this.displayStepFlag
                displayStep(this);
            end
            
            %update Plot.
            if this.plotFlag
                updatePlot(this);
            end
            IsDone = false;
        end















%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
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
            
            this.centroidPlot = plot(this.centroid(1,1), this.centroid(1,2),'ow', 'MarkerFaceColor','red', 'MarkerSize', 8);
            this.centroidPlot.XDataSource = "centroidX";
            this.centroidPlot.YDataSource = "centroidY";


            numConnections = this.numUsers * this.numUavs;
            
            xPointsForPlot = zeros(numConnections*2,1);
            yPointsForPlot = zeros(numConnections*2,1);
            xPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,1);
            yPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,2);
            xPointsForPlot(2:2:end) = this.state.userCoords{1}(:,1);
            yPointsForPlot(2:2:end) = this.state.userCoords{1}(:,2);

            this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");

            circleRadius = mean(this.state.userDistance{1});
            circleTheta = linspace(0,2*pi);
            circleX = circleRadius*cos(circleTheta) + this.state.uavCoords{1}(:,1);
            circleY = circleRadius*sin(circleTheta) + this.state.uavCoords{1}(:,2);
            this.radiusPlot = plot(circleX,circleY,"color","blue");
        end


%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function displayStep(this)
            
            disp ("Step Number:"+this.state.stepCounter{1})
            
            disp("Uav Position:")
            disp("        "+this.state.uavCoords{1}(1)+"    "+this.state.uavCoords{1}(2)+"    "+this.state.uavCoords{1}(3))
            disp("        "+round(this.centroid(1))+"    "+round(this.centroid(2))+"    "+this.centroid(3))


            disp("User Postions:")
            disp(this.state.userCoords{1})
            %{
            disp("K-Means Distribution: (2 uav)")
            if this.numUsers == 1
                numClusters = 1;
            else
                numClusters = 2;
            end
            idx = kmeans([this.state.userCoords{1}(:,1),this.state.userCoords{1}(:,2)],numClusters);
            group1 = [];
            group2 = [];
            for userCtr = 1:this.numUsers
                userGroupAllocation = idx(userCtr);
                if userGroupAllocation == 1
                   group1(end+1,:) = this.state.userCoords{1}(userCtr,:);
                else
                   group2(end+1,:) = this.state.userCoords{1}(userCtr,:);
                end
            end
            disp("Group 1:")
            disp(group1)
            disp("Group 2:")
            disp(group2)

            disp("Reward: ");
            disp("      "+this.rewardGivenForLogging);
            %}
            disp("Pathloss Values: ");
            disp(this.pathlossVector);

            
            disp("Distance Vect:");
            disp(this.state.userDistance{1})

            disp("Heading angles:")
            disp(rad2deg(this.state.userAngleHeading{1}));

            disp("Pitch angles:")
            disp(rad2deg(this.state.userAnglePitch{1}));
            disp(newline)
        end

%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function updatePlot(this)
            %variables must be moved to the base workspace to be used as
            %the data source for plots.
            assignin('base','uavXCoords',this.state.uavCoords{1}(:,1))
            assignin('base','uavYCoords',this.state.uavCoords{1}(:,2))
            assignin('base','userXCoords',this.state.userCoords{1}(:,1))
            assignin('base','userYCoords',this.state.userCoords{1}(:,2))
            assignin('base','centroidX',this.centroid(1,1))
            assignin('base','centroidY',this.centroid(1,2))
            refreshdata(this.uavPlot)
            refreshdata(this.userPlot)
            refreshdata(this.centroidPlot)
            delete(this.connectionPlot)
            delete(this.radiusPlot)

            %number of connection must adapt to include base connections
            %and multiple uavs connections and there allocated users.
            %will change later to work kmeans allocation of users to uavs.
            %baseConnections can be seperate plots.
            numConnections = this.numUsers * this.numUavs;

            xPointsForPlot = zeros(numConnections*2,1);
            yPointsForPlot = zeros(numConnections*2,1);
            xPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,1);
            yPointsForPlot(1:2:end) = this.state.uavCoords{1}(:,2);
            xPointsForPlot(2:2:end) = this.state.userCoords{1}(:,1);
            yPointsForPlot(2:2:end) = this.state.userCoords{1}(:,2);

            this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");

            circleRadius = mean(this.state.userDistance{1});
            circleTheta = linspace(0,2*pi);
            circleX = circleRadius*cos(circleTheta) + this.state.uavCoords{1}(:,1);
            circleY = circleRadius*sin(circleTheta) + this.state.uavCoords{1}(:,2);
            this.radiusPlot = plot(circleX,circleY,"color","blue");

            drawnow;
        end

%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function reward = weightedSumRewardFunction(this)
            reward = 0;
            sortedPathlossVector = sort(this.pathlossVector);
            for pathlossCtr = 1:length(this.pathlossVector)
                reward = reward + sortedPathlossVector(pathlossCtr)*this.pathlossWeights(pathlossCtr);
            end
        end



%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function reward =  standardRewardFunction(this)
                reward = min(this.pathlossVector);
        end


%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function reward = mostPathlossMinusMean(this)
            reward = min(this.pathlossVector) - mean(this.pathlossVector);
        end


%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        function reward = centroidMeanNormalisedRewardFunction(this)
            %needs proper testing as standalone function.
            %user coords data needs to be updated before being called.
            centroid = [mean(this.state.userCoords(:,1)), mean(this.state.userCoords(:,2)), this.uavHeightRel];
            distancesFromCentroid = zeros(1,this.numUsers);
            for userCtr = 1:this.numUsers
                [distancesFromCentroid(1,userCtr),~,~] = yawPitchDistanceFromCoords(centroid, this.state.userCoords(userCtr,:));
            end
            pathlossValuesFromCentroid = pathloss_model(distancesFromCentroid,this.freqHz);
            reward = min(realPathlossValues) - mean(pathlossValuesFromCentroid);
        end

%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    end
end








