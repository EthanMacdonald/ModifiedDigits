for i=1:15
    I = imread(strcat(num2str(i),'.png'));
    I = imresize(I,[48 48],'bilinear');
    imwrite(I,strcat(num2str(i),'.png'));
end