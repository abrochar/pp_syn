addpath ./matlab/export_fig
clear all
%%
N=256;
name = "iso"
if name == "mrw"
    load('./data/data_sixin_paper/demo_mrw2dd_train_N256.mat')
elseif name == "bubble"
    load('./data/data_sixin_paper/demo_brDuD111_N256.mat')
else
    load('./data/data_sixin_paper/ns_randn4_train_N256.mat')
end
img = imgs(:,:,1);
if name == "mrw"
    H=0.2;
    Bxfi = fi_2d_fft(img, H+1);
    vmin = quantile(Bxfi(:),0.01)
    vmax = quantile(Bxfi(:),0.99)
else
    vmin = quantile(img(:),0.01);
    vmax = quantile(img(:),0.99);
end

img = importdata('./results/lrwph_txt/turbiso_rt_ms10kit_9.txt');
if name == "mrw"
    H=0.2
    Bxfi = fi_2d_fft(img,H+1);
    imagesc(Bxfi,[vmin,vmax])
elseif name == "bubble"
    imagesc(img,[vmin,vmax])
else
    img2 = img(N/4:N/4+N/2-1,N/4:N/4+N/2-1);
    imagesc(img2,[vmin,vmax])
end
colormap gray
axis square
set(gca,'XTick',[])
set(gca,'YTick',[])
export_fig('./results/lrwph/turbiso_rt_ms.pdf','-pdf','-transparent')
