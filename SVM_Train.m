imagefiles=dir('D:\Matlab 2017\bin\Dataset\Cervical\*.jpg');
nfiles = length(imagefiles);    % Number of files found
features=zeros(1,14);
for i=1:nfiles
   currentfilename = strcat('D:\Matlab 2017\bin\Dataset\Cervical\',imagefiles(i).name);
   currentimage = imread(currentfilename);
   gray=rgb2gray(currentimage);
   glcm=graycomatrix(gray, 'offset', [0 1], 'Symmetric', true);
   feature_vector=haralickTextureFeatures(glcm,1:14);
   features=[features;feature_vector'];
end
imagefiles=dir('D:\Matlab 2017\bin\Dataset\Lumbar\*.jpg');
mfiles = length(imagefiles);    % Number of files found
for i=1:mfiles
   currentfilename = strcat('D:\Matlab 2017\bin\Dataset\Lumbar\',imagefiles(i).name);
   currentimage = imread(currentfilename);
   gray=rgb2gray(currentimage);
   glcm= graycomatrix(gray, 'offset', [0 1], 'Symmetric', true);
   feature_vector=haralickTextureFeatures(glcm,1:14);
   features=[features;feature_vector'];
end
features(1:1,:)=[];
group = [ones(nfiles,1) ;-1*ones(mfiles,1)];    
SVMStruct = fitcsvm(features, group);
save('SVMStruct');