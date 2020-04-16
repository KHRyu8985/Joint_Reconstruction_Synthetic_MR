%% testing for LORAKS
clear; clc;
load('Data/k_cc_7.mat');
[yRes, xRes, zRes, cRes, dRes, eRes] = size(measc_cc);
%%
kData = squeeze(measc_cc(:,:,10,:,:,1));
%%
R=6;
nacs = 20;
pat = zeros(size(kData)); % initialize sampling mask
pat(:,end/2-nacs/2+1:end/2+nacs/2,:,:,:) = 1;
pat(:,1:R:end,:,1,:) = 1; 
pat(:,round(R/2):R:end,:,2,:) = 1; 
pat(:,1:R:end,:,3,:) = 1; 
pat(:,round(R/2):R:end,:,4,:) = 1; 

kData_us = kData.*pat;
kData_us = reshape(kData_us,yRes,xRes,cRes*dRes);
pat = reshape(pat,yRes,xRes,cRes*dRes);

rank = 500;
recon = AC_LORAKS(kData_us,pat,rank,2,[],[],[],[],100,0);

recon = reshape(recon,yRes,xRes,cRes,dRes);
recon = recon(:,:,:,1);
full_recon = kData(:,:,:,1);

im_comb = sqrt(sum(abs(fftshift(ifft2(ifftshift(recon)))).^2,3))/500000;
full_comb = sqrt(sum(abs(fftshift(ifft2(ifftshift(full_recon)))).^2,3))/500000;

figure; 
subplot(1,2,1); imshow(abs(full_comb));
subplot(1,2,2); imshow(abs(im_comb));
