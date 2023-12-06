%% Question1
clc;
clear;
img = phantom('Modified Shepp-Logan',500);
img_noise=imnoise(img,'gaussian',0,0.0025);
% plot

figure()
subplot(1,2,1);
imshow(img);
title("Phantom image");
subplot(1,2,2)
imshow(img_noise);
title("Phantom noisy image");

%b
out=NLM(img_noise);

% plot

figure()
subplot(1,3,1);
imshow(img);
title("Phantom image");
subplot(1,3,2)
imshow(img_noise);
title("Phantom noisy image");
subplot(1,3,3)
imshow(out);
title("Phantom noisy image after NLM filter");


%c
SNR_nlm=SNR(img,out)
SNR_noise=SNR(img,img_noise)
epi_nlm=EPI(img,out)
epi_noise=EPI(img,img_noise)
%% Question 2
clc;
clear;
img = phantom('Modified Shepp-Logan',500);
img_noise=imnoise(img,'gaussian',0,0.0025);

out=bilateral_filter(img_noise);
% plot

figure()
subplot(1,3,1);
imshow(img);
title("Phantom image");
subplot(1,3,2)
imshow(img_noise);
title("Phantom noisy image");
subplot(1,3,3)
imshow(out);
title("Phantom noisy image after bilateral filter");


%c
SNR_bil=SNR(img,out)
SNR_noise=SNR(img,img_noise)
SNR_bil=EPI(img,out)
epi_noise=EPI(img,img_noise)
%% Question 3
clc;
clear;
img = phantom('Modified Shepp-Logan',500);
img_noise=imnoise(img,'gaussian',0,0.0025);
out=total_variation(img_noise,0.001,10,100);

% plot

figure()
subplot(1,3,1);
imshow(img);
title("Phantom image");
subplot(1,3,2)
imshow(img_noise);
title("Phantom noisy image");
subplot(1,3,3)
imshow(out);
title("Phantom noisy image after total variation with dt=0.001");


%c
SNR_TV=SNR(img,out)
SNR_noise=SNR(img,img_noise)
SNR_TV=EPI(img,out)
epi_noise=EPI(img,img_noise)
%% function
function out=total_variation(img,dt,landa,iteration)

[n,m]=size(img);

delta_pos_x=zeros(n,m);
delta_neg_x=zeros(n,m);
delta_pos_y=zeros(n,m);
delta_neg_y=zeros(n,m);

u_before=img;
for ite=1:iteration
    
       
       delta_pos_x(1:end-1,:)=u_before(2:end,:)-u_before(1:end-1,:);
       delta_neg_x(2:end,:)=u_before(2:end,:)-u_before(1:end-1,:);
       delta_pos_y(:,1:end-1)=u_before(:,2:end)-u_before(:,1:end-1);
       delta_neg_y(:,2:end)=u_before(:,2:end)-u_before(:,1:end-1);
       
       m_x=((sign(delta_pos_x.*u_before)+sign(delta_neg_x.*u_before))/2).*(min(abs(delta_pos_x.*u_before),abs(delta_neg_x.*u_before)));
       m_y=((sign(delta_pos_y.*u_before)+sign(delta_neg_y.*u_before)/2)).*(min(abs(delta_pos_y.*u_before),abs(delta_neg_y.*u_before)));
       
       num1=delta_pos_x.*u_before;
       num2=delta_pos_y.*u_before;
       denum1=sqrt((delta_pos_x.*u_before).^2+m_y.^2+eps);
       denum2=sqrt((delta_pos_y.*u_before).^2+m_x.^2+eps);
       temp1=num1./denum1;
       temp2=num2./denum2;
       final_temp1=zeros(n,m);
       final_temp2=zeros(n,m);
       final_temp1(2:end,:)=temp1(2:end,:)-temp1(1:end-1,:);
       final_temp2(:,2:end)=temp2(:,2:end)-temp2(:,1:end-1);
       
       u_new=u_before+dt*(final_temp1+final_temp2)+landa*dt*(img-u_before);
       
       
        
     
    % boundary conditions
    u_new(1,:)=u_new(2,:);
    u_new(end,:)=u_new(end-1,:);
    u_new(:,1)=u_new(:,2);
    u_new(:,end)=u_new(:,end-1);
    
    u_before=u_new;
end
out=u_before;

end
function out=NLM(img)

[n,m]=size(img);
hv=0.1;
Gp= fspecial('gaussian',3,hv);
Gp = Gp / sum(sum(Gp));
 
out=zeros(n,m);
for i=5:1:n-5
    for j=5:1:m-5
        sum1=0;
        denum=0;
       
        dist=0;
        wmax=0;
        for k=-3:1:3
            for s=-3:1:3
           if(k==0 && s==0) 
               continue; 
           end
           W_org=img(i-1:i+1,j-1:j+1);
           W=img(i+k-1:i+k+1,j+s-1:j+s+1);
           dist=((W_org-W).*(W_org-W));
           dist_Gp=Gp.*dist;
           dist_final=sum(sum(dist_Gp));
           patch= exp(-(dist_final)/(hv*hv));
           sum1=sum1+(img(i+k,j+s).*patch);
           %denum=denum+sum(patch,'all');
           denum=denum+patch;
           if dist_final>wmax                
                    wmax=dist_final;                   
                end
            end
        end
            sum1=sum1+wmax*img(i,j);
            denum=denum+wmax;
        if denum > 0
            out(i,j)=sum1/denum;
        else
            out(i,j) = img(i,j);
        end       
        
    end
    
end
%figure()
%imshow(out);
end
function snr = SNR( X , Y)
x2 = X .* X;
temp1 = sum(x2,'all');
x_y =( X - Y).^2;
temp2 = sum(x_y , 'all');
snr = 10 * log( temp1 / temp2)/log(10);
end
function out=bilateral_filter(img)

[n,m]=size(img);
hx=0.1;
hv=0.1;
%serch window
h=10;
out=zeros(n,m);
for i=12:1:n-12
    for j=12:1:m-12
        sum1=0;
        denum=0;
       
     
        for k=-h:1:h
            for s=-h:1:h
           if(i<=h+2)
               i1=i+1;
           else
               i1=k;
           end
           if(j<=h+2)
               j1=j+1;
           else
               j1=s;
           end
           if(i>=n-h-2)
               i1=i-(n-h-2);
           else
               i1=k;
           end
           if(j>=m-h-2)
               j1=j-(m-h-2);
           else
               j1=s;
           end
           
           W_org=img(i-1:i+1,j-1:j+1);
           W=img(i+i1-1:i+i1+1,j+j1-1:j+j1+1);
           
           %G_hg
           dist=((W_org-W).*(W_org-W));
           dist_G_hg=sum(sum(dist));
           G_hg= exp(-(dist_G_hg)/(2*hv*hv));
           
           %G_hx
           
           dist_x=(i1^2+j1^2)*ones(3) ;
           dist_G_hx=sum(sum(dist));
           G_hx= exp(-(dist_G_hg)/(2*hx*hx));
           
           sum1=sum1+(img(i+i1,j+j1).*(G_hx* G_hg));
           %denum=denum+sum(patch,'all');
           denum=denum+G_hx* G_hg;
 
            end
        end
           
        if denum > 0
            out(i,j)=sum1/denum;
        else
            out(i,j) = img(i,j);
        end       
        
    end
    
end
end