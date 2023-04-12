function [LOS_map, dist_map, radio_horizon, FresLoss_map, d1_dist_map, d1_height_map] = get_LOS_maps(elev_map, tx_coord_x_vect, tx_coord_y_vect, x_vector, y_vector,...
    tx_msl_height, rx_height, map_resolution,freq_Hz, KED_flag, user_scanbox_limits_idx)

    obstacleInputs = 1;
    if ~isempty(user_scanbox_limits_idx)
        X1 = user_scanbox_limits_idx(1);
        X2 = user_scanbox_limits_idx(2);
        Y1 = user_scanbox_limits_idx(3);
        Y2 = user_scanbox_limits_idx(4);
    else
        X1 = 1;
        X2 = length(x_vector);
        Y1 = 1;
        Y2 = length(y_vector);
    end
    
    LOS_map = zeros(length(x_vector), length(y_vector)) + 2;
    dist_map = zeros(length(x_vector), length(y_vector));
    FresLoss_map = zeros(length(x_vector), length(y_vector));
    d1_dist_map = zeros(length(x_vector), length(y_vector),obstacleInputs);
    d1_height_map = zeros(length(x_vector), length(y_vector),obstacleInputs);
    
    %for X = 1:length(x_vector)
    %    for Y = 1:length(y_vector)
    %        if (X>=X1)&&(X<=X2)&&(Y>=Y1)&&(Y<=Y2)
    for X = X1:X2
        for Y = Y1:Y2
            [obstacle, Path_array_ind, Path_distance, PL_diff, TD] = get_LOS_train([tx_coord_x_vect tx_coord_y_vect], [x_vector(X) y_vector(Y)], elev_map, x_vector, y_vector, ...
                tx_msl_height, rx_height, map_resolution, freq_Hz, KED_flag);
            if sum(obstacle)>0
                LOS_map(X,Y) = 0;            
                %first_obstacle_ind = find(obstacle,1);
                %LOS_map(Path_array_ind(1:first_obstacle_ind)) = 1; % Areas with LOS
                %LOS_map(Path_array_ind(first_obstacle_ind:end)) = 0; % Areas with no LOS
            else
                LOS_map(X,Y) = 1; % Areas with LOS            
                %LOS_map(Path_array_ind) = 1; % Areas with LOS
            end
    
            if KED_flag
                if isempty(TD.d1Seg)
                    d1_dist_map(X,Y,:) = nan;
                    d1_height_map(X,Y,:) = nan;
                else
                    % [~,jj] = min(TD.ElevDiff60Seg); % Take the largest obstacle
                    [~,jj] = sort(TD.ElevDiff60Seg);
                    d1_dist_map(X,Y,1:min(length(jj),obstacleInputs)) = TD.d1Seg(jj(1:min(length(jj),obstacleInputs)));
                    d1_height_map(X,Y,1:min(length(jj),obstacleInputs)) = TD.ElevDiff60Seg(jj(1:min(length(jj),obstacleInputs)));
                end
            end
    
            %dist_map(Path_array_ind) = Path_distance;
            dist_map(X,Y) = Path_distance(end);
            %FresLoss_map(Path_array_ind) = PL_diff;
            FresLoss_map(X,Y) = sum(PL_diff);
        end
    end
    %end
    
    radio_horizon = sqrt(((6371000 + (tx_msl_height)).^2 - (6371000)^2)); % LOS horizon distance, ignoring elevation in between
    radio_horizon_map = dist_map < radio_horizon;
    
    LOS_map = (LOS_map & radio_horizon_map);
    disp(LOS_map)
