classdef MultiAgentMistralRLEnvClass 
    properties

        %% Notes
        % ~ Move Flags to env constructor and set them after calling
        %   super constructor?
        % ~ Integrate UserMobility class.
        % ~ Integrate dynamic map resolution.
        % ~ Calculate d1 dist in reset function.

        plotFlag = true;
        Figure = false;
        uavPlot;
        userPlot;
        centroidPlot;
        connectionPlots;
        uavUserAllocation;
        centroidDistance;
        centroidAngleHeading;
        centroidAnglePitch;
        uavAllocationIndex;
        uavCoords;
        userCoords;
        max_P_Rx_map;
        

        numUsers;
        numUavs = 1;
        
        uavHeightRel = 100;
        userHeightRel = 1.5;
        KEDFlag = true;
        freqHz = 160e6;
        rewardGivenForLogging;

        mapResolution = 400;
        forestCoverOneHot;
        elevMap;
        xVector;
        yVector;

        centroids;
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
function this = MultiAgentMistralRLEnvClass(numUavs, numUsers)
            this.numUavs = numUavs;
            this.numUsers = numUsers;
            this = reset(this);
        end

        function this = runSim(this, episodeCount, stepCount)
            load("agent2Long_Trained.mat");

            for episodeCtr = 1:episodeCount
                this = reset(this);
                for stepCtr = 1:stepCount
                    actionList = zeros(1,this.numUavs);
                    for uavCtr = 1:this.numUavs
                        [distance,heading,pitch] = yawPitchDistanceFromCoords(this.uavCoords(uavCtr,:),this.centroids(uavCtr,:));
                        obs = {};
                        obs{1} = distance;
                        obs{2} = heading;
                        obs{3} = pitch;

                        disp(obs{1})
                        disp(obs{2})
                        disp(obs{3})
                        disp(obs)
                        act = getAction(agent2Long_Trained,obs);
                        actionList(uavCtr) = act{1};
                    end
                    this = stepFunc(this,actionList);
                end
            end
        end





%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function this = reset(this)
            [this.forestCoverOneHot, this.elevMap, this.xVector, this.yVector ] = generateTerrain(this.mapResolution);


            this.uavCoords = zeros(this.numUavs,3);
            
            for uavCtr = 1:this.numUavs
                if this.randomUavPositionFlag
                    uavCoordsTemp = randi([0 71],1,2);
                    uavCoordsTemp = [uavCoordsTemp(1)*this.mapResolution uavCoordsTemp(2)*this.mapResolution this.uavHeightRel ];
                else
                    uavCoordsTemp = [4000,4000,this.uavHeightRel];
                end
                this.uavCoords(uavCtr,:) = uavCoordsTemp;
            end

            if this.randomUserPositionFlag
                userCoordsInd = randi([1,70], this.numUsers,2);
                this.userCoords = userCoordsInd * this.mapResolution;
                this.userCoords(:,3) = 0;
                for userCtr = 1:this.numUsers
                    userIndexX = userCoordsInd(userCtr,1);
                    userIndexY = userCoordsInd(userCtr,2);
                    this.userCoords(userCtr,3) = this.elevMap(userIndexX,userIndexY);
                end
                this.userCoords(:,3) = this.userCoords(:,3)+this.userHeightRel;
            else
                this.userCoords = zeros(this.numUsers,3);
                this.userCoords(:,1) = 24000;
                this.userCoords(:,2) = 24000;
                this.userCoords(:,3) = this.elevMap(61,61);
                this.userCoords(:,3) = this.userCoords(:,3)+this.userHeightRel;
            end
            

            userDistance = zeros(this.numUavs,this.numUsers);
            userAngleHeading = zeros(this.numUavs,this.numUsers);
            userAnglePitch = zeros(this.numUavs,this.numUsers);
            for uavCtr = 1:this.numUavs
                for userCtr = 1:this.numUsers
                    [userDistance(uavCtr,userCtr), userAngleHeading(uavCtr,userCtr), userAnglePitch(uavCtr,userCtr)] = yawPitchDistanceFromCoords(this.uavCoords(uavCtr,:), this.userCoords(userCtr,:));
                end
            end
            
            
            this.uavUserAllocation = zeros(this.numUavs,this.numUsers);
            

            this.uavAllocationIndex = kmeans([this.userCoords(:,1),this.userCoords(:,2)],this.numUavs);
            for userCtr = 1:this.numUsers
                this.uavUserAllocation(this.uavAllocationIndex(userCtr),userCtr) = 1;
            end

            disp(this.uavUserAllocation);
            disp("-")
            disp(this.uavAllocationIndex);

            this.centroids = zeros(this.numUavs,3);
            
            %{
            for uavCtr = 1:this.numUavs
                xSum = 0;
                ySum = 0;
                usersAllocated = 0;
                for userCtr = 1:this.numUsers
                    if this.uavUserAllocation(uavCtr,userCtr)
                       xSum = xSum + this.userCoords(userCtr,1);
                       ySum = ySum + this.userCoords(userCtr,2);
                       usersAllocated = usersAllocated+1;
                    end
                end
                this.centroids(uavCtr,:) = [(xSum / usersAllocated), ySum / usersAllocated, this.uavHeightRel];
            end
            %}
            %{
            disp(this.uavCoords)
            disp(this.userCoords)
            for uavCtr = 1:this.numUavs
                allocatedUserCoords = zeros(1,3);
                allocatedUserDistances = zeros(1,1);
                indCounter = 1;
                for userCtr = 1:this.numUsers
                    if this.uavUserAllocation(uavCtr,userCtr)
                        allocatedUserCoords(indCounter,:) = this.userCoords(userCtr,:);
                        
                        allocatedUserDistances(uavCtr,indCounter) = userDistance(uavCtr,userCtr);
                        indCounter = indCounter+1;
                    end
                end
                disp("-")
                [~,grInd] = max(userDistance(uavCtr,:));
                disp(grInd)
                disp(userDistance(uavCtr,:))
                disp(allocatedUserCoords(1,grInd))
                this.centroids(uavCtr,:) = allocatedUserCoords(1,grInd);
                disp(this.centroids)

                
            end
            %}
            for uavCtr = 1:this.numUavs
                usersAllocated = 0;
                uavAllocatedUserCoords = zeros(1,3);
                allocatedUserDistances = zeros(1,1);
                for userCtr = 1:this.numUsers
                    if this.uavAllocationIndex(userCtr,1) == uavCtr
                        uavAllocatedUserCoords(usersAllocated+1,:) = this.userCoords(userCtr,:);
                        [allocatedUserDistances(1,usersAllocated+1),~,~] = yawPitchDistanceFromCoords(this.uavCoords(uavCtr,:),this.userCoords(userCtr,:));
                        usersAllocated = usersAllocated + 1;
                    end
                end
                [~,grInd] = max(allocatedUserDistances(1,:));
                this.centroids(uavCtr,:) = uavAllocatedUserCoords(grInd,:);
                this.centroids(uavCtr,3) = this.uavHeightRel;
                disp(usersAllocated);
                disp(uavAllocatedUserCoords);
            end




            this.centroidDistance = zeros(1,this.numUavs);
            this.centroidAngleHeading = zeros(1,this.numUavs);
            this.centroidAnglePitch = zeros(1,this.numUavs);
            for uavCtr = 1:this.numUavs
                [this.centroidDistance(1,uavCtr), this.centroidAngleHeading(1,uavCtr), this.centroidAnglePitch(1,uavCtr)] = yawPitchDistanceFromCoords(this.uavCoords(uavCtr,:),this.centroids(uavCtr,:));
            end




            %for plotting - taken from original mistral script.
            P_Rx_map = zeros(size(this.elevMap,1),size(this.elevMap,2),this.numUavs);
            this.max_P_Rx_map = max(P_Rx_map,[],3);
            
            %plot initial environment.
            if this.plotFlag
                this = plot(this);
            end
        end






%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function this = stepFunc(this,ActionList)
            disp("step activated")

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
            for uavCtr = 1:this.numUavs
                ActionIndex = ActionList(:,uavCtr);
                Action = moveMatrix(ActionIndex,:);
                Action = [Action(1) Action(2) 0];
                % Update Uav Coordinates
                this.uavCoords(uavCtr,:) = this.uavCoords(uavCtr,:) + Action;
                
                %enforce border on movement.
                if this.uavCoords(uavCtr,1) > 28400
                    this.uavCoords(uavCtr,1) = 28400;
                elseif this.uavCoords(uavCtr,1) <  0
                    this.uavCoords(uavCtr,1) = 0;
                end 
            
                if this.uavCoords(uavCtr,2) > 28000
                    this.uavCoords(uavCtr,2) = 28000;
                elseif this.uavCoords(uavCtr,2) <  0
                    this.uavCoords(uavCtr,2) = 0;
                end
            end

           

            nextUserCoords = this.userCoords;

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
            this.userCoords = nextUserCoords;
        
            %{
            for uavCtr = 1:this.numUavs
                xSum = 0;
                ySum = 0;
                usersAllocated = 0;
                for userCtr = 1:this.numUsers
                    if this.uavUserAllocation(uavCtr,userCtr)
                       xSum = xSum + this.userCoords(userCtr,1);
                       ySum = ySum + this.userCoords(userCtr,2);
                       usersAllocated = usersAllocated+1;
                    end
                end
                this.centroids(uavCtr,:) = [(xSum / usersAllocated), ySum / usersAllocated, this.uavHeightRel];
            end
            %}
            for uavCtr = 1:this.numUavs
                usersAllocated = 0;
                uavAllocatedUserCoords = zeros(1,3);
                allocatedUserDistances = zeros(1,1);
                for userCtr = 1:this.numUsers
                    if this.uavAllocationIndex(userCtr,1) == uavCtr
                        uavAllocatedUserCoords(usersAllocated+1,:) = this.userCoords(userCtr,:);
                        [allocatedUserDistances(1,usersAllocated+1),~,~] = yawPitchDistanceFromCoords(this.uavCoords(uavCtr,:),this.userCoords(userCtr,:));
                        usersAllocated = usersAllocated + 1;
                    end
                end
                [~,grInd] = max(allocatedUserDistances(1,:));
                this.centroids(uavCtr,:) = uavAllocatedUserCoords(grInd,:);
                this.centroids(uavCtr,3) = this.uavHeightRel;
                disp(usersAllocated);
                disp(uavAllocatedUserCoords);
            end

            if this.plotFlag
                this = updatePlot(this);
            end

        end















%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
        function this = plot(this)
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
            P_Rx_map_plot = surf(this.xVector,this.yVector,this.max_P_Rx_map');
            
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

            this.uavPlot = plot(this.uavCoords(:,1), this.uavCoords(:,2), '^w', 'MarkerFaceColor','k', 'MarkerSize', 10);
            this.uavPlot.XDataSource = "uavXCoords";
            this.uavPlot.YDataSource = "uavYCoords";

            this.userPlot = plot(this.userCoords(:,1), this.userCoords(:,2), 'ow', 'MarkerFaceColor','k', 'MarkerSize', 10);
            this.userPlot.XDataSource = "userXCoords";
            this.userPlot.YDataSource = "userYCoords";
                        
            this.centroidPlot = plot(this.centroids(:,1), this.centroids(:,2),'ow', 'MarkerFaceColor','red', 'MarkerSize', 6);
            this.centroidPlot.XDataSource = "centroidX";
            this.centroidPlot.YDataSource = "centroidY";


            
            this.connectionPlots = [];
            for uavCtr = 1:this.numUavs
                xPointsForPlot = [];
                yPointsForPlot = [];
                for userCtr = 1:this.numUsers
                    if (this.uavUserAllocation(uavCtr,userCtr))
                        xPointsForPlot(end+1) = this.uavCoords(uavCtr,1);
                        yPointsForPlot(end+1) = this.uavCoords(uavCtr,2);
                        xPointsForPlot(end+1) = this.userCoords(userCtr,1);
                        yPointsForPlot(end+1) = this.userCoords(userCtr,2);
                        %this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");
                    end
                end
                this.connectionPlots(end+1) = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");
            end
            
            %{
            circleRadius = mean(this.state.userDistance{1});
            circleTheta = linspace(0,2*pi);
            circleX = circleRadius*cos(circleTheta) + this.state.uavCoords{1}(:,1);
            circleY = circleRadius*sin(circleTheta) + this.state.uavCoords{1}(:,2);
            this.radiusPlot = plot(circleX,circleY,"color","blue");
            %}
        end


%% --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        function displayStep(this)
            clc;
            disp ("Step Number:"+this.state.stepCounter{1})
            
            disp("Uav Position:")
            disp("        "+this.state.uavCoords{1}(1)+"    "+this.state.uavCoords{1}(2)+"    "+this.state.uavCoords{1}(3))
            disp("        "+round(this.centroids))


            disp("User Postions:")
            disp(this.state.userCoords{1})
            
            %{
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
        function this = updatePlot(this)
            %variables must be moved to the base workspace to be used as
            %the data source for plots.
            assignin('base','uavXCoords',this.uavCoords(:,1))
            assignin('base','uavYCoords',this.uavCoords(:,2))
            assignin('base','userXCoords',this.userCoords(:,1))
            assignin('base','userYCoords',this.userCoords(:,2))
            assignin('base','centroidX',this.centroids(:,1))
            assignin('base','centroidY',this.centroids(:,2))
            refreshdata(this.uavPlot)
            refreshdata(this.userPlot)
            refreshdata(this.centroidPlot)
            delete(this.connectionPlots)

            %delete(this.radiusPlot)

            this.connectionPlots = [];
            for uavCtr = 1:this.numUavs
                xPointsForPlot = [];
                yPointsForPlot = [];
                for userCtr = 1:this.numUsers
                    if (this.uavUserAllocation(uavCtr,userCtr))
                        xPointsForPlot(end+1) = this.uavCoords(uavCtr,1);
                        yPointsForPlot(end+1) = this.uavCoords(uavCtr,2);
                        xPointsForPlot(end+1) = this.userCoords(userCtr,1);
                        yPointsForPlot(end+1) = this.userCoords(userCtr,2);
                        %this.connectionPlot = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");
                    end
                end
                this.connectionPlots(end+1) = plot(xPointsForPlot,yPointsForPlot,"LineWidth",1,"color","black");
            end
            
            %{
            circleRadius = mean(this.state.userDistance{1});
            circleTheta = linspace(0,2*pi);
            circleX = circleRadius*cos(circleTheta) + this.state.uavCoords{1}(:,1);
            circleY = circleRadius*sin(circleTheta) + this.state.uavCoords{1}(:,2);
            this.radiusPlot = plot(circleX,circleY,"color","blue");
            %}
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








