clear all
root = './data_ns_randn4_N512';
K=110;

for k=1:K
    [vor_time,enstrophy,time_list] = OMEGA_1_nopen_ns_randn4_N512();
    save(sprintf('%s/omega_us_nondipole_sin_%s.mat',root,datestr(now,'YYYYmmDD_HHMMSS')),'vor_time','enstrophy','time_list');
end

lst = dir(root);
vordata = cell(1,K);

imgs_all=zeros(256,256,K);
for i=3:3+K-1
    load(sprintf('%s/%s',root,lst(i).name))
    vor=vor_time(:,:,end);
    rvor = imresize(vor,0.5);
    imgs_all(:,:,i-2)=rvor;
    close all
end

% rescale
imgs = imgs_all(:,:,1:10);
save('ns_randn4_train_N256.mat','imgs')

imgs = imgs_all(:,:,11:end);
save('ns_randn4_test_N256.mat','imgs')

im=imgs(:,:,1);
std(im(:))
imagesc(im); colormap gray;
spIm = mySpectre2D(abs(fft2(im)));
plot(log10(spIm))

