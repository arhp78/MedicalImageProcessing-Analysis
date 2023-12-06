clc
img=imread("heart_ct.jpg");
J = imnoise(img,'salt & pepper',0.5);
imshow(J)
%%
img1 = imread('heart_ct_S&P-0.01.jpg');
img2 = imread('heart_ct_S&P-0.02.jpg');
img3 = imread('heart_ct_S&P-0.2.jpg');

multi = cat(1,img1,img2,img3);


montage(multi);
%%
img1 = imread('heart_ct_S&P-0.001.jpg');
img2 = imread('heart_ct_S&P-0.01.jpg');
img3 = imread('heart_ct_S&P-0.02.jpg');
img4 = imread('heart_ct_S&P-0.2.jpg');
img5 = imread('heart_ct_S&P-0.5.jpg');
figure
ax1 = subplot(2,2,1);
imshow(img2)
title('heartct S&P with 0.01')

ax2 = subplot(2,2,2);
imshow(img3)
title('heartct S&P with 0.02')

ax3 = subplot(2,2,3);
imshow(img4)
title('heartct S&P with 0.2')

ax4 = subplot(2,2,4);
imshow(img5)
title('heartct S&P with 0.5')
%%
img1 = imread('heart_ct_gaussian.jpg');
img2 = imread('heart_ct_gaussian_m=0_var=0.01.jpg');
img3 = imread('heart_ct_gaussian_m=0_var=0.1.jpg');
img4 = imread('heart_ct_gaussian_m=0_var=1.jpg');
img5 = imread('heart_ct_gaussian_m=1_var=0.1.jpg');
figure
ax1 = subplot(2,2,1);
imshow(img2)
title('heart ct gaussian m=0 var=0.01')

ax2 = subplot(2,2,2);
imshow(img3)
title('heart ct gaussian m=0 var=0.1')

ax3 = subplot(2,2,3);
imshow(img4)
title('heart ct gaussian m=0 var=1')

ax4 = subplot(2,2,4);
imshow(img5)
title('heart ct gaussian m=1 var=0.1')
%%
mean_3 = (1/9).*ones(3);
mean_5 = (1/25).*ones(5);
mean_7 = (1/49).*ones(7);
img1 = imread('heart_ct_gaussian_m=0_var=0jpg');

filtered_img3=imfilter(img1 , mean_3,'conv');
filtered_img5=imfilter(img1 , mean_5,'conv');
filtered_img7=imfilter(img1 , mean_7,'conv');
figure
ax1 = subplot(2,2,1);
imshow(img1)
title('heart ct S&P with density 0.02 ')

ax2 = subplot(2,2,2);
imshow(filtered_img3)
title('heart ct S&P with density 0.02  filtered by 3*3')

ax3 = subplot(2,2,3);
imshow(filtered_img5)
title('heart ct S&P with density 0.02  filtered by 5*5')

ax4 = subplot(2,2,4);
imshow(filtered_img7)
title('heart ct S&P with density 0.02  filtered by 7*7')

