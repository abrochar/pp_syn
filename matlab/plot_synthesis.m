addpath ./altmany-export_fig-4703a84
clear all
mroot = './cerfeuil_synthesis/';
proot = './pdfs/';

name = 'anisotur2a_modelD';
N=256;

% these are 1% and 99% quantile of pixel values of barx
vmin_tur2a=-1.8366047044206786;
vmax_tur2a=1.8366047044206786;
vmin_mrw2dd=-0.0475;
vmax_mrw2dd= 0.0475;
vmin_bubbles=-0.25124018357711797;
vmax_bubbles=0.394454;
vmin_anisotur2a=-1.7441073493983232;
vmax_anisotur2a=1.7441073493983232;

switch name
    case 'tur2a_ori'
        imfn = '../data/ns_randn4_train_N256.mat';
        ks=1;
        vmin = vmin_tur2a;
        vmax = vmax_tur2a;
    case 'tur2a_modelA'
        imfn = 'maxent_synthesis_tur2a_kb1.mat';
        ks=1;
        vmin = vmin_tur2a;
        vmax = vmax_tur2a;
    case 'tur2a_modelB'
        imfn = 'tur2a_modelB_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_tur2a;
        vmax = vmax_tur2a;
    case 'tur2a_modelC'
        imfn = 'tur2a_modelC_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_tur2a;
        vmax = vmax_tur2a;
    case 'tur2a_modelD'
        imfn = 'tur2a_modelD_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_tur2a;
        vmax = vmax_tur2a;
    case 'mrw2dd_ori'
        imfn = '../data/demo_mrw2dd_train_N256.mat';
        ks=1;
        vmin = vmin_mrw2dd;
        vmax = vmax_mrw2dd;
    case 'mrw2dd_modelA'
        imfn = 'maxent_synthesis_mrw2dd_kb1.mat';
        ks=1;
        vmin = vmin_mrw2dd;
        vmax = vmax_mrw2dd;
    case 'mrw2dd_modelB'
        imfn = 'mrw2dd_modelB_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_mrw2dd;
        vmax = vmax_mrw2dd;
    case 'mrw2dd_modelC'
        imfn = 'mrw2dd_modelC_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_mrw2dd;
        vmax = vmax_mrw2dd;
    case 'mrw2dd_modelD'
        imfn = 'mrw2dd_modelD_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_mrw2dd;
        vmax = vmax_mrw2dd;
    case 'bubbles_ori'
        imfn = '../data/demo_brDuD111_N256.mat';
        ks=1;
        vmin = vmin_bubbles;
        vmax = vmax_bubbles;
    case 'bubbles_modelA'
        imfn = 'maxent_synthesis_bubbles_kb1.mat';
        ks=1;
        vmin = vmin_bubbles;
        vmax = vmax_bubbles;
    case 'bubbles_modelB'
        imfn = 'bubbles_modelB_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_bubbles;
        vmax = vmax_bubbles;
    case 'bubbles_modelC'
        imfn = 'bubbles_modelC_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_bubbles;
        vmax = vmax_bubbles;
    case 'bubbles_modelD'
        imfn = 'bubbles_modelD_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_bubbles;
        vmax = vmax_bubbles;    
    case 'anisotur2a_ori'
        imfn = '../data/ns_randn4_aniso_train_N256.mat';
        ks=1;
        vmin = vmin_anisotur2a;
        vmax = vmax_anisotur2a;
    case 'anisotur2a_modelA'
        imfn = 'maxent_synthesis_anisotur2a_kb1.mat';
        ks=1;
        vmin = vmin_anisotur2a;
        vmax = vmax_anisotur2a;        
    case 'anisotur2a_modelB'
        imfn = 'anisotur2a_modelB_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_anisotur2a;
        vmax = vmax_anisotur2a;
    case 'anisotur2a_modelC'
        imfn = 'anisotur2a_modelC_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_anisotur2a;
        vmax = vmax_anisotur2a;
    case 'anisotur2a_modelD'
        imfn = 'anisotur2a_modelD_synthesis_ks0.mat';
        ks=1;
        vmin = vmin_anisotur2a;
        vmax = vmax_anisotur2a;
end
load(sprintf('%s/%s',mroot,imfn))
if size(imgs,1)==N
    img = imgs(:,:,ks);
else
    img = reshape(imgs(ks,:,:),N,N);
end

if contains(name,'ori')
    vmin_ = quantile(img(:),.01);
    vmax_ = quantile(img(:),.99);
    fprintf('vmin=%g,vmax=%g\n',vmin_,vmax_)
end

if contains(name,'mrw2dd')
    H = 0.2;
    %---fractionally integrate
    Bxfi = fi_2d_fft(img,H+1);
    % the abs max of vmin and vmax defines the cmax range
    %vmin = quantile(Bxfi(:),0.01);
    %vmax = quantile(Bxfi(:),0.99);
    imagesc(Bxfi,[vmin,vmax])
elseif contains(name,'anisotur2a')
    img2 = img(N/4:N/4+N/2-1,N/4:N/4+N/2-1); % img for figs
    imagesc(img2,[vmin,vmax])
else
    imagesc(img,[vmin,vmax])
end
colormap gray
axis square
set(gca,'XTick',[])
set(gca,'YTick',[])
export_fig(sprintf('%s/%s.pdf',proot,name),'-pdf','-transparent')
