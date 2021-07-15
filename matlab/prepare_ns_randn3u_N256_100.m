clear all
root = './data_ns_randn3u_N256';
K=100;

for k=1:K
    [vor_time,enstrophy,time_list] = OMEGA_1_nopen_ns_randn3_N256();
    save(sprintf('%s/omega_us_nondipole_sin_%s.mat',root,datestr(now,'YYYYmmDD_HHMMSS')),'vor_time','enstrophy','time_list');
end

lst = dir(root);
vordata = cell(1,K);

imgs=zeros(256,256,K);
for i=3:3+K-1
    load(sprintf('%s/%s',root,lst(i).name))
    vor=vor_time(:,:,end);
    imgs(:,:,i-2)=vor;
    close all
end

% substract mean and scale into [-0.5,0.5]
imu=mean(imgs(:));
imgs=imgs-imu;
imax=max(imgs(:));
imin=min(imgs(:));
ibox=max(imax,-imin);
imgs=imgs/(2*ibox);

save('ns_randn3u_N256.mat','imgs')

im=imgs(:,:,1);
std(im(:))
imagesc(im); colormap gray;