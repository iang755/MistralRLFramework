% Solution derived from mathematics found here : https://stackoverflow.com/questions/58469297/how-do-i-calculate-the-yaw-pitch-and-roll-of-a-point-in-3d
function [distance, angleHeading, anglePitch] = yawPitchDistanceFromCoords(originXYZ, destinationXYZ)
    distance = pdist([originXYZ;destinationXYZ], 'euclidean');

    angleHeading = atan2(destinationXYZ(3) - originXYZ(3),destinationXYZ(1) - originXYZ(1));
    
    opposite = destinationXYZ(2) - originXYZ(2);

    adjacent = sqrt((destinationXYZ(1) - originXYZ(1))^2 + (destinationXYZ(3) - originXYZ(3))^2);

    anglePitch = atan2(opposite, adjacent);

end
