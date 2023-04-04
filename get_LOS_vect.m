%function [LOS_map, dist_map, radio_horizon] = get_LOS_vect(elev_map, tx_coord_x_vect, tx_coord_y_vect, x_vector, y_vector, tx_msl_height, rx_height, map_resolution)
function [LOS_vect, dist_vect,radio_horizon, PL_diff_vect, d1_dist_vect, d1_height_vect,forestOnehot_vect] = get_LOS_vect(elev_map, forestCover_onehot, tx_coord_x_vect, tx_coord_y_vect, rx_coord_x_vect, rx_coord_y_vect,...
    tx_msl_height, rx_height, map_resolution, x_vector, y_vector, freq_Hz, KED_flag)



obstacleInputs = 1;
num_tx_locations = length(tx_coord_x_vect);
num_rx_locations = length(rx_coord_x_vect);
LOS_vect = zeros(num_tx_locations, num_rx_locations);
dist_vect = zeros(num_tx_locations, num_rx_locations);
PL_diff_vect = zeros(num_tx_locations, num_rx_locations);
d1_dist_vect = zeros(num_tx_locations, num_rx_locations,obstacleInputs);
d1_height_vect = zeros(num_tx_locations, num_rx_locations,obstacleInputs);
forestOnehot_vect = zeros(num_tx_locations, num_rx_locations,3);
%LOS_map = zeros(length(x_vector), length(y_vector)) + 2;
%dist_map = zeros(length(x_vector), length(y_vector));

for tx_ctr = 1:num_tx_locations
    for rx_ctr = 1:num_rx_locations        
        %rx_coord = [x_vector(X) y_vector(Y)];
        %tx_coord = [tx_coord_x_vect tx_coord_y_vect];
        [obstacle, Path_array_ind, Path_distance, PL_diff,TD] = get_LOS_train([tx_coord_x_vect(tx_ctr) tx_coord_y_vect(tx_ctr)], [rx_coord_x_vect(rx_ctr) rx_coord_y_vect(rx_ctr)],...
            elev_map, x_vector, y_vector, tx_msl_height, rx_height, map_resolution, freq_Hz, KED_flag);
        disp("sum(obstacle):")
        disp(sum(obstacle))
        if sum(obstacle)>0 % There are obstacles. Set to 2 to ignore "edge" blockages.
            LOS_vect(tx_ctr, rx_ctr) = 0;
            %first_obstacle_ind = find(obstacle,1);
            %LOS_map(Path_array_ind(1:first_obstacle_ind)) = 1; % Areas with LOS
            %LOS_map(Path_array_ind(first_obstacle_ind:end)) = 0; % Areas with no LOS
        else
            LOS_vect(tx_ctr, rx_ctr) = 1;
            %LOS_map(Path_array_ind) = 1; % Areas with LOS
        end

         if isempty(TD.d1Seg)
            d1_dist_vect(tx_ctr,rx_ctr,:) = nan;
            d1_height_vect(tx_ctr,rx_ctr,:) = nan;
        else            
           % [~,jj] = min(TD.ElevDiff60Seg); % Take the largest obstacle
            [~,jj] = sort(TD.ElevDiff60Seg);  
            d1_dist_vect(tx_ctr,rx_ctr,1:min(length(jj),obstacleInputs)) = TD.d1Seg(jj(1:min(length(jj),obstacleInputs)));
            d1_height_vect(tx_ctr,rx_ctr,1:min(length(jj),obstacleInputs)) = TD.ElevDiff60Seg(jj(1:min(length(jj),obstacleInputs)));
        end
        %dist_map(Path_array_ind) = Path_distance;
        dist_vect(tx_ctr, rx_ctr) = Path_distance(end);
        PL_diff_vect(tx_ctr, rx_ctr) = sum(PL_diff);
        forestOnehot_vect(tx_ctr,rx_ctr,:)=forestCover_onehot(round(rx_coord_x_vect(rx_ctr)/map_resolution)+1, round(rx_coord_y_vect(rx_ctr)/map_resolution)+1,:);
        
    end
end

radio_horizon = sqrt(((6371000 + (tx_msl_height)).^2 - (6371000)^2)); % LOS horizon distance, ignoring elevation in between

radio_horizon_vect = dist_vect > radio_horizon;
disp("Radio Horizon Vect:")
disp(radio_horizon_vect)
LOS_vect = (LOS_vect | radio_horizon_vect);
disp("LOS Vect:")
disp(LOS_vect)
