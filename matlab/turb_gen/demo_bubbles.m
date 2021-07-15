sid='D111';

im=imread(['./' sid '.gif']);
figure(1); imagesc(im); colormap gray; colorbar

oimg=double(im)./255;

br=4; % border to cut off
oimg = oimg(1+br:end-br,1+br:end-br);
N2=size(oimg,1);
% oimg=(oimg-mean(oimg(:)))/std(oimg(:));
% oimg=oimg-mean(oimg(:));
M=max(max(oimg(:)),-min(oimg(:)));
oimg=oimg./(2*M);

N=256;K=floor(N2/N)^2;
imgs=zeros(N,N,K);
offset = floor((N2-N)/(sqrt(K)-1));
assert(N2>N)
k = 1;
for k1=1:floor(N2/N)
    for k2=1:floor(N2/N)
        ox=offset*(k1-1)+1; 
        oy=offset*(k2-1)+1;
        img_=oimg(ox:ox+N-1,oy:oy+N-1);
        img_=img_-mean(img_(:)); % each one has a zero mean
        fprintf('ox=%d,oy=%d\n',ox,oy);
        imgs(:,:,k)=img_;
        imagesc(imgs(:,:,k)); colormap gray; colorbar
        k=k+1;
    end
end

figure; imagesc(log(abs(fftshift(fft2(imgs(:,:,1))))+eps)); colormap gray; colorbar
std(img_(:))

save(['./demo_bubbles' (sid) '_N256.mat'],'imgs')

%% downsample
% small_imgs = zeros(N/2,N/2,K);
% for k = 1:K
%     small_imgs(:,:,k)=imresize(imgs(:,:,k),0.5);
% end
% imgs=small_imgs;
% save(['../src/cache/rawdata/demo_brDu' (sid) '_N128.mat'],'imgs')
