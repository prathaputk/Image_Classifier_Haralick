load('SVMStruct')
imagefiles=dir('D:\Matlab 2017\bin\Dataset\Test\Cervical\*.jpg');
nfiles = length(imagefiles);
features=zeros(1,14);
for i=1:nfiles
   currentfilename = strcat('D:\Matlab 2017\bin\Dataset\Test\Cervical\',imagefiles(i).name);
   currentimage = imread(currentfilename);
   gray=rgb2gray(currentimage);
   glcm= graycomatrix(gray, 'offset', [0 1], 'Symmetric', true);
   feature_vector=haralickTextureFeatures(glcm,1:14);
   features=[features;feature_vector'];
end
imagefiles=dir('D:\Matlab 2017\bin\Dataset\Test\Lumbar\*.jpg');
mfiles = length(imagefiles);    % Number of files found
for i=1:mfiles
   currentfilename = strcat('D:\Matlab 2017\bin\Dataset\Test\Lumbar\',imagefiles(i).name);
   currentimage = imread(currentfilename);
   gray=rgb2gray(currentimage);
   glcm= graycomatrix(gray, 'offset', [0 1], 'Symmetric', true);
   feature_vector=haralickTextureFeatures(glcm,1:14);
   features=[features;feature_vector'];
end
features(1:1,:)=[];
result=predict(SVMStruct,features);
Cervical_result=[result(1:nfiles,:)];
Lumbar_result=[result(nfiles+1:nfiles+mfiles,:)];
Cervical_score=sum(Cervical_result(:)==1)/nfiles
Lumbar_score=sum(Lumbar_result(:)==-1)/mfiles

   %radial and feedforward