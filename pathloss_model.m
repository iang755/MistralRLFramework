%Original Author: Lester Ho, Tyndall National Institute.

function PL_map = pathloss_model(dist_map, freq_Hz, LOS_map)


alpha_LOS = 2.13; %PL exponent
%alpha_LOS = 2.6; %PL exponent
alpha_NLOS = 4; %PL exponent


if nargin == 2
    PL_map = -((alpha_LOS.*10).*log10(dist_map) + (alpha_LOS.*10)*log10(freq_Hz) + (alpha_LOS.*10)*log10((4*pi)/3e8)); % Path loss
else
    PL_map = zeros(size(LOS_map));
    PL_map(LOS_map) = -((alpha_LOS.*10).*log10(dist_map(LOS_map)) + (alpha_LOS.*10)*log10(freq_Hz) + (alpha_LOS.*10)*log10((4*pi)/3e8)); % Path loss
    PL_map(~LOS_map) = -((alpha_NLOS.*10).*log10(dist_map(~LOS_map)) + (alpha_NLOS.*10)*log10(freq_Hz) + (alpha_NLOS.*10)*log10((4*pi)/3e8)); % Path loss
end

%PL_map(LOS_map==1) = -((alpha_LOS.*10).*log10(dist_map) + (alpha_LOS.*10)*log10(freq_Hz) + (alpha_LOS.*10)*log10((4*pi)/3e8)); % Path loss

