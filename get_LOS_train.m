function [obstacle, Path_array_ind, Path_distance, PL_diff, TD] = get_LOS_train(tx_coord, rx_coord, elev_map, x_vector, y_vector,tx_rel_height, rx_rel_height, map_resolution_m, freq_Hz, KED_flag)
txrx_horiz_distance = hypot(tx_coord(1)-rx_coord(1), tx_coord(2)-rx_coord(2));
m = (tx_coord(2)-rx_coord(2))/(tx_coord(1)-rx_coord(1));
%if abs(m)>1e6
%    m = inf;
%end
R = round(txrx_horiz_distance/map_resolution_m)+1;
if R<2 %txrx_horiz_distance==0 % If UAV is directly above user
    obstacle = 0;
    tx_coord_ind = round(tx_coord./map_resolution_m)+1;
    if tx_rel_height<elev_map(tx_coord_ind(1),tx_coord_ind(2))
        tx_rel_height = elev_map(tx_coord_ind(1),tx_coord_ind(2)) + 10;
    end
    Path_distance = tx_rel_height-(rx_rel_height + elev_map(round(rx_coord(1)/map_resolution_m)+1, round(rx_coord(2)/map_resolution_m)+1));
    Path_array_ind = [];
    PL_diff = 0;
    TD.d1Seg = [];
    TD.ElevDiff60Seg = [];
    TD.ElevDiffSeg = [];
    TD.PLDiffSeg = [];
    TD.D = Path_distance(end);
    TD.PL = pathloss_model(TD.D, freq_Hz);
    return
end
if isinf(m)
    if tx_coord(1)==rx_coord(1) % If vertical line
        %yy = linspace2(tx_coord(2),rx_coord(2),R);
        yy = (rx_coord(2)-tx_coord(2))/(R-1)*(0:R-1) + tx_coord(2);
        xx = zeros(1,R)+tx_coord(1);
    elseif tx_coord(2)==rx_coord(2) % If horizontal line
        %xx = linspace2(tx_coord(1),rx_coord(1),R);
        xx = (rx_coord(1)-tx_coord(1))/(R-1)*(0:R-1) + tx_coord(1);
        yy = zeros(R,1)+tx_coord(2);
    end
else
    %xx = linspace2(tx_coord(1),rx_coord(1),R);
    xx = (rx_coord(1)-tx_coord(1))/(R-1)*(0:R-1) + tx_coord(1); % Faster than linspace
    c = tx_coord(2)-m*tx_coord(1);
    yy = m.*xx+c;
end
Path_ind_x = round(xx/map_resolution_m)+1;
Path_ind_y = round(yy/map_resolution_m)+1;
dist_from_rx = sqrt((x_vector(Path_ind_x)-rx_coord(1)).^2 + (y_vector(Path_ind_y)-rx_coord(2)).^2);
Path_array_ind = Path_ind_x + (Path_ind_y-1)*size(elev_map,1);
Path_elev = elev_map(Path_array_ind);
% if UAV_flag==1
%     if tx_rel_height < Path_elev(1)
%         disp('TX is assumed to be UAV, but height is below ground, setting height to 10m above ground');
%         tx_real_height = tx_rel_height + 10;
%     else
%         tx_real_height = tx_rel_height;
%     end
% else
%     tx_real_height = Path_elev(1) + tx_rel_height;
% end
if tx_rel_height < Path_elev(1)
    %disp('TX height is below ground, setting height to 10m above ground');
    tx_real_height = Path_elev(1) + 10;
else
    tx_real_height = tx_rel_height;
end
rx_real_height = Path_elev(end) + rx_rel_height;
tx2rx_angle_deg = atand((dist_from_rx(1))/(tx_real_height-rx_real_height));

rx_path_height = Path_elev+rx_rel_height;
rel_path_height = tx_real_height - rx_path_height;
dist_from_tx = flip(dist_from_rx);
Path_distance = sqrt(rel_path_height.^2 + dist_from_tx.^2);
% Fresnel zone calculation
h_rel = tx_real_height-rx_real_height; % Relative height between UAV and user
D = max(dist_from_rx);
theta = atan(h_rel/D); % User to UAV angle
d_user2ob = D-dist_from_tx;%+1; % Ground distance between obstruction and UAV
d_sl = h_rel/sin(theta); % straight line distance between user and UAV
d1 = d_user2ob./cos(theta); % straight line distance between user and obstruction
d2 = d_sl-d1; % straight line distance between obstruction and UAV
d2(1) = 0; d2(end)=d_sl;
d1(end) = 0; d1(1)=d_sl;
lambda = 3e8/freq_Hz;
Fres_radius = sqrt( lambda.*(d1.*d2)./(d1 + d2) ); % First fresnel zone radius
Path_z = (dist_from_rx)./tand(tx2rx_angle_deg)+ rx_real_height;
obstacle = (Path_elev>(Path_z-0.6.*Fres_radius)); % Only consider it if blockage is within 60% of Fresnel zone radius
elev_diff_mod = (Path_z-0.6.*Fres_radius) - Path_elev;
elev_diff = Path_z - Path_elev;

TD.D = d_sl;
TD.PL = pathloss_model(TD.D, freq_Hz);
if nnz(obstacle)>0
    if KED_flag 
        d1(d1==0) = 0.01; % To avoid inf when doing Fresnel integration
        d2(d2==0) = 0.01;
        PL_diff = zeros(1,length(Path_distance));
        dx = diff([0,obstacle]);
        segStart = find(dx > 0); % Start of any blockage segments
        segEnd = find(dx < 0)-1; % End of any blockage segments
        TD.d1Seg = zeros(size(segStart));
        TD.ElevDiff60Seg = zeros(size(segStart));
        TD.ElevDiffSeg = zeros(size(segStart));
        TD.PLDiffSeg = zeros(size(segStart));

        for segCtr = 1:length(segStart)
            segInd = segStart(segCtr):segEnd(segCtr);
            d_ob2uav = dist_from_tx(segInd);%+1;
            Delta = ((Path_elev(segInd)-tx_real_height).*D + (tx_real_height-rx_real_height).*d_ob2uav)./...
                (sqrt(D.^2 + (tx_real_height-rx_real_height).^2));
            v = Delta.*( sqrt( (2./lambda).*(1./d1(segInd) + 1./d2(segInd)) ));
            %v(v<-100)=-100;
            %v(v>100)=100;
            [Cv, Sv] = fresnelCS(v);
            PL_diff_seg = -20.*log10(sqrt((1-Cv-Sv).^2 + (Cv-Sv).^2)./2);
            %PL_diff_seg(abs(v)>5) = 0;
            [aa,bb] = max(PL_diff_seg);
            PL_diff(segInd(bb)) = aa;
            TD.d1Seg(segCtr) = d1(segInd(bb));
            TD.ElevDiff60Seg(segCtr) = elev_diff_mod(segInd(bb));
            TD.ElevDiffSeg(segCtr) = elev_diff(segInd(bb));
            TD.PLDiffSeg(segCtr) = PL_diff(segInd(bb));            
        end
        %obstacle = zeros(1,length(Path_distance)); % Set obstacles to zero, since NLOS calculations are based on diffraction
    else
        PL_diff = zeros(1,length(Path_distance));
    end
else
    PL_diff = zeros(1,length(Path_distance));
    TD.d1Seg = [];
    TD.ElevDiff60Seg = [];
    TD.ElevDiffSeg = [];
    TD.PLDiffSeg = [];
end
