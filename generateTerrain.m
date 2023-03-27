function [forestCoverOneHot, elevMap, xVector, yVector ] = generateTerrain(map_resolution)
    load('TestMaps2.mat', 'elev_map'); 
    load('TestMaps2.mat', 'x_vector', 'y_vector');    
    if map_resolution ~= x_vector(2)-x_vector(1) 
        [elevMap, xVector, yVector] = interpolate_map(elev_map, map_resolution, x_vector, y_vector); 
    end
    %Actual forestCover
    load("forestCover.mat",'forestCover');
    forestCover = flipud(forestCover);
    forestCover = forestCover';

    for_vector_x = 0:(xVector(end)/(size(forestCover,1)-1)):xVector(end);
    for_vector_y = 0:(yVector(end)/(size(forestCover,2)-1)):yVector(end);
    [forestCover,~,~]=interpolate_map(forestCover,map_resolution,for_vector_x,for_vector_y);
    forestCover = round(forestCover);

    foo = forestCover;
    foo(foo==0)=1;
    forestCoverOneHot = zeros(length(xVector),length(yVector),3); 
    for x = 1:length(xVector)
        for y = 1:length(yVector)
            forestCoverOneHot(x,y,foo(x,y))=1;
        end
    end
end

