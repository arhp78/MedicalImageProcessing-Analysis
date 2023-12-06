%97101507
%% Question 1
clc;
clear;
img_org = imread("city_orig.jpg");
img_noise = imread("city_noise.jpg");
img_org=double(img_org);
img_noise=double(img_noise);
%part one : top left: 1:end/2   salt&paper
%part one : bottom left: end/2 : end  salt&paper +gaussian
%part one : bottom right: 1:end/2 gaussian  
%part one : top right
%SNR before filtering
fprintf("before filtering ")
snr_salt_before=SNR(img_org(1:530/2,1:750/2),img_noise(1:530/2,1:750/2))
snr_both_before=SNR(img_org(530/2+1:end,1:750/2),img_noise(530/2+1:end,1:750/2))
snr_gauss_before=SNR(img_org(530/2+1:end,750/2+1:end),img_noise(530/2+1:end,750/2+1:end))
snr_without_before=SNR(img_org(1:530/2,750/2+1:end),img_noise(1:530/2,750/2+1:end))
%SNR after apply median filter
fprintf("after apply median filter ")
img_median=medfilt2(img_noise);
snr_salt_median=SNR(img_org(1:530/2,1:750/2),img_median(1:530/2,1:750/2))
snr_both_median=SNR(img_org(530/2+1:end,1:750/2),img_median(530/2+1:end,1:750/2))
snr_gauss_median=SNR(img_org(530/2+1:end,750/2+1:end),img_median(530/2+1:end,750/2+1:end))
snr_without_median=SNR(img_org(1:530/2,750/2+1:end),img_median(1:530/2,750/2+1:end))
figure
imshow(uint8(img_median));
title(" after apply median filter ");
%SNR after apply gauss filter
fprintf("after apply gauss filter ")
img_gauss=imgaussfilt(img_noise,2);
snr_salt_median=SNR(img_org(1:530/2,1:750/2),img_gauss(1:530/2,1:750/2))
snr_both_median=SNR(img_org(530/2+1:end,1:750/2),img_gauss(530/2+1:end,1:750/2))
snr_gauss_median=SNR(img_org(530/2+1:end,750/2+1:end),img_gauss(530/2+1:end,750/2+1:end))
snr_without_median=SNR(img_org(1:530/2,750/2+1:end),img_gauss(1:530/2,750/2+1:end))

figure
imshow(uint8(img_gauss));
title(" after apply gauss filter with sigma=2 ");

%SNR after apply box filter
fprintf("after apply box filter ")
img_box=imboxfilt(img_noise,5);
snr_salt_box=SNR(img_org(1:530/2,1:750/2),img_box(1:530/2,1:750/2))
snr_both_box=SNR(img_org(530/2+1:end,1:750/2),img_box(530/2+1:end,1:750/2))
snr_gauss_box=SNR(img_org(530/2+1:end,750/2+1:end),img_box(530/2+1:end,750/2+1:end))
snr_without_box=SNR(img_org(1:530/2,750/2+1:end),img_box(1:530/2,750/2+1:end))
figure
imshow(uint8(img_box));
title(" after apply 5*5 box filter ");
%% Question 2
%section a
clc;
clear;
img = imread("hand_xray.jpg");
img=double(img);
img_fft=fftshift(fft2(img));
power_fft = sum(sum(abs(img_fft).^2));
figure()
imshow(uint8(15*log(img_fft)))
ylabel("dB");
title(" FFT of image ");

%section b
meanIntensity_image = mean(img(:))
img_fft=fft2(img);
power_img = sum(sum(img.^2)); 

meanIntensity_fft = img_fft(1,1)*power_img/power_fft
%% Question 3
img = imread("chessboard.jpg");
img_grey=rgb2gray(img);
img_edge = edge(img_grey,'Canny');
img_edge1 = edge(img_grey,'sobel');
img_edge2 = edge(img_grey,'log');

subplot(1,3,1), imshow(img_edge),title(" canny ");
subplot(1,3,2), imshow(img_edge1),title(" sobel ");
subplot(1,3,3), imshow(img_edge2),title(" laplacian of gussian ");

%% Question 4
clc;
clear;
img = imread("hand_xray.jpg");
img_fft=fft2(double(img));
img_rotate=ifft2(conj(img_fft));
figure
imshow(uint8(img_rotate));
%% Question 5
clc;
clear;
img1 = imread("hand_xray.jpg");
img2 = imread("brain_xray.jpg");

fft_result1=fft2(double(img1));
mag1=abs(fft_result1);                           
phase1=angle(fft_result1);

fft_result2=fft2(double(img2));
mag2=abs(fft_result2);                           
phase2=angle(fft_result2);

img12=ifft2(mag1.*exp(1i*phase2));
img22=ifft2(mag2.*exp(1i*phase1));

% plot
figure()
subplot(2,2,1)
imshow(uint8(img1));
title("image1");
subplot(2,2,2)
imshow(uint8(img2));
title("image2");
subplot(2,2,3)
imshow(uint8(img12));
title(" magnitude of image1+ phase of image2 ");
subplot(2,2,4)
imshow(uint8(img22));
title(" magnitude of image2+ phase of image1 ");
%% Question 6
%section a
clc;
clear;
img = imread("wall.jpg");
img=double(img);
img_fft=fftshift(fft2(img));
figure()
imshow(uint8(15*log(img_fft)))
ylabel("dB");
title(" FFT of image ");

%b
%first calculate x & y center image
[y1,x1,z1] = size(img);
x_center=int32(x1/2);
y_center=int32(y1/2);

lpf = zeros(y1 , x1);
lpf(y_center-15:y_center+15,x_center-15:x_center+15)=1;
img_fft_filtred=lpf.*img_fft;
img_filter=ifft2(ifftshift(img_fft_filtred));
img_fft_filtred=(1-lpf).*img_fft;
img_filter_hf=ifft2(ifftshift(img_fft_filtred));
figure()
subplot(1,2,1), imshow(uint8(img_filter)),title("  image filtered after apply LPF ");
subplot(1,2,2), imshow(uint8(img_filter_hf)),title(" image filtered after apply HPF ");
%d
% GAUUSIAN FILTER
img_fft=fftshift(fft2(img));
x=251;
y=253;
filter_gauss= GaussianFilter(y,x, 10);

img_fft_filtred=filter_gauss.*img_fft;
img_filter=ifft2(ifftshift(img_fft_filtred));

img_fft_filtred_HF=(1-filter_gauss).*img_fft;
img_filter_hf=ifft2(ifftshift(img_fft_filtred_HF));
figure()
subplot(1,2,1), imshow(uint8(img_filter)),title("  image filtered after apply gaussian LPF ");
subplot(1,2,2), imshow(uint8(img_filter_hf)),title(" image filtered after apply gaussian HPF ");

% Laplacian FILTER
img_fft=fftshift(fft2(img));
x=251;
y=253;
filter_laplace= laplacian(253,251);

img_fft_filtred=filter_laplace.*img_fft;
img_filter=ifft2(ifftshift(img_fft_filtred));

img_fft_filtred_HF=(1 - filter_laplace).*img_fft;
img_filter_hf=ifft2(ifftshift(img_fft_filtred_HF));
img_filter_hf=(img_filter_hf-min(img_filter_hf,[],'all'))./(max(img_filter_hf,[],'all')-min(img_filter_hf,[],'all'));
img_filter_hf=img_filter_hf* 200;
figure()
subplot(1,2,1), imshow(uint8(img_filter)),title("  image filtered after apply Laplacian LPF ");
subplot(1,2,2), imshow(uint8(img_filter_hf)),title(" image filtered after apply Laplacian HPF ");

% Butterworth filter
a=1;   b=1 ;
%%Generate a distance matrix based on the size of the input image.

dist = distmatrix(253,251);
%%Generate a Butterworth high-pass filter.
cutoff = 500; order = 1;
H_but = 1 ./ (1 + (cutoff ./ dist) .^ (2 * order));
%%Apply high-frequency emphasis filtering to the high-pass filter(butter) and display the resulting filter.
H_but_hfe = a + (b .* H_but);
img_filtered=real(ifft2(ifftshift(img_fft.*H_but_hfe)));
img_filtered_lf=real(ifft2(ifftshift(img_fft.*(1-H_but_hfe./(max(H_but_hfe,[],'all'))))));

figure()
subplot(1,2,1),imshow(uint8(img_filtered)),title(" image filtered after apply butterworth LPF ");
subplot(1,2,2),  imshow(uint8(200*img_filtered_lf)),title("  image filtered after apply Laplacian HPF ");



%%
%function
function snr = SNR( X , Y)
x2 = X .* X;
sum1 = sum(x2,'all');
x_y =( X - Y).^2;
sum2 = sum(x_y , 'all');
snr = 10 * log10( sum1 / sum2);
end

function filter_gauss= GaussianFilter(nRows, nCols, sigma)
    if mod(nRows,2)==0
       centerI =int32(nRows/2)-1;
    else
       centerI =int32(nRows/2);
    end
         if mod(nCols,2) == 0
             centerJ =int32(nCols/2)-1;
         else
             centerJ =int32(nCols/2);
          end
       filter_gauss=zeros(nRows,nCols);
    centerJ=double(centerJ);
    centerI=double(centerI);
    for j = 1:nCols 
     for i =1:nRows
       g = exp(-1.0 * ((i - centerI).^ 2 + (j - centerJ).^2) / (2 * sigma.^2));
       filter_gauss(i,j)=g;
     end
    end
end

function filter_laplac= laplacian(nRows, nCols)
filter_laplac=zeros(nRows,nCols);

    for i = 1:nRows
            for j = 1:nCols
		    lap= (i-nRows/2)^2 + (j-nCols/2)^2;
		    filter_laplac(i,j) = -1 * lap; % high pass
            end
    end
    filter_laplac=(filter_laplac-min(filter_laplac,[],'all'))./(max(filter_laplac,[],'all')-min(filter_laplac,[],'all'));
end
function y = distmatrix(M,N)

u = 0:(M - 1);
v = 0:(N - 1);

ind_u = find(u > M/2);
u(ind_u) = u(ind_u) - M;
ind_v = find(v > N/2);
v(ind_v) = v(ind_v) - N;

[V, U] = meshgrid(v, u);

%calculate distance matrix
y = sqrt((U .^ 2) + (V .^ 2));
end