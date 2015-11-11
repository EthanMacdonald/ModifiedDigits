FOLDER = 'train_images/';
OUTPUT_FOLDER = 'enhanced_train_images/';
EMBOSSING = [-1 0 ; 0 1];
NUMBER_OF_SAMPLES = 20000;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

outputsImg = zeros(NUMBER_OF_SAMPLES,1 + 48*48);
outputLabels = zeros(NUMBER_OF_SAMPLES,1 + 1);

for i = 1:1
    [d ,numberOfImages] = size(images);
    
    indx = floor(rand()*numberOfImages);

    I = reshape(images(:,indx),[28,28]);
    I = uint8(I .* double(255));
    
    R = imresize(I,1.714,'bilinear');
 
    rotate = floor(rand()*360);
    B = imrotate(R,rotate,'bilinear','crop');
    
    pat = floor(rand()*14) + 1;
    tex = imread(strcat(strcat('patterns/',num2str(pat)),'.png'));

    R = rgb2gray(tex) + B(1:48,1:48);

    R = conv2(double(-EMBOSSING-EMBOSSING.'),double(R));
    
    R = mat2gray(R(1:48,1:48));
    out = reshape(R,[1,48*48]);
    
    figure;
    imshow(R)
    
    outputsImg(i,1) = i;
    outputsImg(i,2:48*48+1) = out;
    
    outputLabels(i,1) = i;
    outputLabels(i,2) = labels(indx);
    
    imwrite(R,strcat(strcat(OUTPUT_FOLDER,num2str(i)),'.png'));
end

%csvwrite('enhanced_training_inputs.csv',outputsImg);
%csvwrite('enhanced_training_outputs.csv',outputLabels);

%zip('enhanced_train_images.zip',OUTPUT_FOLDER);