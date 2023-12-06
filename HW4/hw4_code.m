%% Question 1
% section a
clc ; clear;
pic1 = imread("Mri1.bmp");
pic2 = imread("Mri2.bmp");
pic3 = imread("Mri3.bmp");
pic4 = imread("Mri4.bmp");
pic5 = imread("Mri5.bmp");
[n,m]=size(pic1);
%image1 

img=zeros(n,m,3);
img(:,:,2)=pic1;
img(:,:,1)=pic2;
img(:,:,3)=pic3;

%image2 
img1=zeros(n,m,3);
img1(:,:,3)=pic1 ;
img1(:,:,2)=pic4;
img1(:,:,1)=pic5;

%plot
figure()
subplot(1,2,1);
imshow(uint8(img));
title("image 1");
subplot(1,2,2)
imshow(uint8(img1));
title("image 2");
%% section b
clc;clear;
pic1 = imread("Mri2.bmp");
B = reshape(pic1,1,[]);
Ph = 5;
nclus = 4;
[centers,U] = fcm(double(B), nclus, Ph);
maxU = max(U);
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);
index3 = find(U(3,:) == maxU);
index4 = find(U(4,:) == maxU);
%img_fin1 = reshape(centers(index1,:),256,256);
%img_fin1=50;


img_fin2 = reshape(centers(2,:),256,256);
img_fin3 = reshape(centers(3,:),256,256);
img_fin4 = reshape(centers(4,:),256,256);
figure()
subplot(1,4,1);
imshow(uint8(img_fin1));
subplot(1,4,2);
imshow(uint8(img_fin2));
subplot(1,4,3);
imshow(uint8(img_fin3));
subplot(1,4,4);
imshow(uint8(img_fin4));

%% section c
clc;clear;
pic1 = imread("Mri1.bmp");
k=4;
L = imsegkmeans(pic1,k);
B = labeloverlay(pic1,L);
imshow(B);


%% Question 2
% section a - GVF
clc ; clear;
pic1 = imread("F:\uni\T6\MIAP\HW4\Blur1.png");
pic2 = imread("F:\uni\T6\MIAP\HW4\Blur2.png");
mu=0.5;
ITER=10;
[u1,v1] = GVF(pic1, mu, ITER);
[u2,v2] = GVF(pic2, mu, ITER);
figure()
subplot(1,2,1);
imshow(u1);
title("GVF blur1 with mu=0.5")
subplot(1,2,2);
imshow(u2);
title("GVF blur2 with mu=0.5")
%% Question 3
