classdef MistralRLEnvClass < rl.env.MATLABEnvironment
    properties
        state;

        plotFlag;
        Figure = false;
        uavPlot;
        userPlot;
        connectionPlot;
        borderPlot;

        numUsers;
        numUavs = 1;        
        uavHeightRel = 1000;
        userHeightRel = 1.5;
        baseHeightRel = 20;
        KEDFlag = true;
        freqHz = 160e6;

        mapResolution = 400;
        forestCoverOneHot;
        elevMap;
        xVector;
        yVector;

        furthestUserPlot;
        furthestUserCoords;

        radiusPlot;
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
            
            obsInfo(1) = rlNumericSpec([1 1]);
            obsInfo(2) = rlNumericSpec([1 1]);

            obsInfo(3) = rlNumericSpec([1 1]);
            obsInfo(4) = rlNumericSpec([1 1]);
            obsInfo(5) = rlNumericSpec([1 1]);
            obsInfo(6) = rlNumericSpec([1 1]);

            obsInfo(7) = rlNumericSpec([1 1]);
            obsInfo(8) = rlNumericSpec([1 1]);
            obsInfo(9) = rlNumericSpec([1 1]);
            obsInfo(10) = rlNumericSpec([1 1]);

            obsInfo(11) = rlNumericSpec([1 1]);
            obsInfo(12) = rlNumericSpec([1 1]);
            obsInfo(13) = rlNumericSpec([1 1]);
            obsInfo(14) = rlNumericSpec([1 1]);

            this = this@rl.env.MATLABEnvironment(obsInfo,actionInfo);


            this.plotFlag = plotFlag;
            this.numUsers = 3;
            %this.numUsers = numUsers;

            validateEnvironment(this)
        end







%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function [InitialObservation, LoggedSignal] = reset(this)
            [this.forestCoverOneHot, this.elevMap, this.xVector, this.yVector ] = generateTerrain(this.mapResolution);
            baseCoords = [13200 13200 0];

            if this.randomUavPositionFlag
                uavCoords = randi([2 69],1,2);
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
            

            [~,~,~, PL_diff_vect, ~,~,~] = get_LOS_vect(this.elevMap, this.forestCoverOneHot, uavCoords(:,1), uavCoords(:,2), userCoords(:,1),userCoords(:,2), ...
                this.uavHeightRel,this.userHeightRel,this.mapResolution,this.xVector,this.yVector,this.freqHz, true );
            plVect = zeros(1,this.numUsers);
            for userCtr = 1:this.numUsers
                plVect(userCtr) = pathloss_model(userDistance(1,userCtr),this.freqHz,1);
            end

            [~, furthestUserIndices] = sort(userDistance,'descend');

            InitialObservation{1} = round(uavCoords(1,1)/1600);
            InitialObservation{2} = round(uavCoords(1,2)/1600);
            
            furthestUserIndex = furthestUserIndices(1,1);

            this.furthestUserCoords = userCoords(furthestUserIndex,:);
            this.furthestUserCoords(1,3) = this.uavHeightRel;

            InitialObservation{3} = round(userCoords(furthestUserIndex,1)/1600);
            InitialObservation{4} = round(userCoords(furthestUserIndex,2)/1600);
            InitialObservation{5} = -PL_diff_vect(1,furthestUserIndex);
            InitialObservation{6} = plVect(1,furthestUserIndex);

            secondFurthestUserIndex = furthestUserIndices(1,2);
            InitialObservation{7} = round(userCoords(secondFurthestUserIndex,1)/1600);
            InitialObservation{8} = round(userCoords(secondFurthestUserIndex,2)/1600);
            InitialObservation{9} = -PL_diff_vect(1,secondFurthestUserIndex);
            InitialObservation{10} = plVect(1,secondFurthestUserIndex);
            
            thirdFurthestUserIndex = furthestUserIndices(1,3);
            InitialObservation{11} = round(userCoords(thirdFurthestUserIndex,1)/1600);
            InitialObservation{12} = round(userCoords(thirdFurthestUserIndex,2)/1600);
            InitialObservation{13} = -PL_diff_vect(1,thirdFurthestUserIndex);
            InitialObservation{14} = plVect(1,thirdFurthestUserIndex);

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
            elseif nextUavCoords(1) <  400
                nextUavCoords(1) = 400;
            end 
        
            if nextUavCoords(2) > 28000
                nextUavCoords(2) = 28000;
            elseif nextUavCoords(2) <  400
                nextUavCoords(2) = 400;
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

            [~,~,~, blockagePlVect, nextUserD1Distance,hVect,~] = get_LOS_vect(this.elevMap, this.forestCoverOneHot, nextUavCoords(:,1), nextUavCoords(:,2), nextUserCoords(:,1),nextUserCoords(:,2), ...
                this.uavHeightRel,this.userHeightRel,this.mapResolution,this.xVector,this.yVector,this.freqHz, true );
        
            LoggedSignals.uavCoords{1} =        nextUavCoords;
            LoggedSignals.userCoords{1} =       nextUserCoords;
            LoggedSignals.userDistance{1} =     nextUserDistance;
            LoggedSignals.userAngleHeading{1} = nextUserAngleHeading;
            LoggedSignals.userAnglePitch{1} =   nextUserAnglePitch;
            LoggedSignals.userD1Distance{1} =   nextUserD1Distance;
            
            distancePlVect = zeros(1,this.numUsers);
            for userCtr = 1:this.numUsers
                distancePlVect(userCtr) = pathloss_model(nextUserDistance(1,userCtr),this.freqHz, 1);
            end

            [~, furthestUserIndices] = sort(nextUserDistance,'descend');

            
            NextObs{1} = round(nextUavCoords(1,1)/1600);
            NextObs{2} = round(nextUavCoords(1,2)/1600);
            
            furthestUserIndex = furthestUserIndices(1,1);

            this.furthestUserCoords = nextUserCoords(furthestUserIndex,:);
            this.furthestUserCoords(1,3) = this.uavHeightRel;



            NextObs{3} = round(nextUserCoords(furthestUserIndex,1)/1600);
            NextObs{4} = round(nextUserCoords(furthestUserIndex,2)/1600);
            NextObs{5} = -blockagePlVect(1,furthestUserIndex);
            NextObs{6} = distancePlVect(1,furthestUserIndex);
            
            secondFurthestUserIndex = furthestUserIndices(1,2);
            NextObs{7} = round(nextUserCoords(secondFurthestUserIndex,1)/1600);
            NextObs{8} = round(nextUserCoords(secondFurthestUserIndex,2)/1600);
            NextObs{9} = -blockagePlVect(1,secondFurthestUserIndex);
            NextObs{10} = distancePlVect(1,secondFurthestUserIndex);
            
            thirdFurthestUserIndex = furthestUserIndices(1,3);
            NextObs{11} = round(nextUserCoords(thirdFurthestUserIndex,1)/1600);
            NextObs{12} = round(nextUserCoords(thirdFurthestUserIndex,2)/1600);
            NextObs{13} = -blockagePlVect(1,thirdFurthestUserIndex);
            NextObs{14} = distancePlVect(1,thirdFurthestUserIndex);


            this.state = LoggedSignals;          

            realPathlossVector = distancePlVect - blockagePlVect;

            Reward = min(realPathlossVector);
            if isnan(Reward)
                Reward = -100;
            end

            clc;
            disp ("Step Number:"+this.state.stepCounter{1})
            disp("observation");
            disp(NextObs);
            disp("uav coords");
            disp(nextUavCoords);
            disp("user coords");
            disp(nextUserCoords);
            disp("distance");
            disp(nextUserDistance);
            disp("d1 distance");
            disp(nextUserD1Distance);
            disp("d1 height");
            disp(hVect)
            disp("pl diff ");
            disp(blockagePlVect);
            disp("Reward")
            disp(Reward);
            
          
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
            
            this.furthestUserPlot = plot(this.furthestUserCoords(1,1), this.furthestUserCoords(1,2),'ow', 'MarkerFaceColor','red', 'MarkerSize', 8);
            this.furthestUserPlot.XDataSource = "furthestUserX";
            this.furthestUserPlot.YDataSource = "furthestUserY";


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
        function updatePlot(this)
            %variables must be moved to the base workspace to be used as
            %the data source for plots.
            assignin('base','uavXCoords',this.state.uavCoords{1}(:,1))
            assignin('base','uavYCoords',this.state.uavCoords{1}(:,2))
            assignin('base','userXCoords',this.state.userCoords{1}(:,1))
            assignin('base','userYCoords',this.state.userCoords{1}(:,2))
            assignin('base','furthestUserX',this.furthestUserCoords(1,1))
            assignin('base','furthestUserY',this.furthestUserCoords(1,2))
            refreshdata(this.uavPlot)
            refreshdata(this.userPlot)
            refreshdata(this.furthestUserPlot)
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

    end
end








