addpath ./export_fig

% compute radial power spectrum
name = 'tur2a';

switch name
    case 'tur2a'
        droot = './maxent_out/';
        load('./sixin/ns_randn4_train_N256.mat')
        tkt = 'pwregress_maxent_bumps2d_dj0_nor_Delta2_tur2a_J5_L8_K10_m1';
end

%load(sprintf('%s/%s.mat',droot,tkt));

N = size(imgs,1);
K = size(imgs,3);

spImgs = zeros(N,N,K);
for k=1:K
    spImgs(:,:,k)=(abs(fft2(imgs(:,:,k))).^2)/(N^2);
end
hatK0 = mean(spImgs,3);
spK0 = mySpectre2D(hatK0);

%%
nbins = 1;
Bloss=zeros(nbins,1);
BentX=zeros(nbins,1);
BentXrec=zeros(nbins,1);
for kb=nbins:nbins
    load(sprintf('%s/%s_kb%d.mat',droot,tkt,kb));
    spXrec = mySpectre2D(hXrec);
    
    figure;
    xk = 0:pi/(N/2):pi;
    xk = xk(2:end);
    plot(xk,log10(spK0),'b-')
    hold on
    plot(xk,log10(spXrec),'b--')

    set(gca,'fontsize',24)
    xticks([0,pi/4,pi/2,pi/4*3,pi])
    xticklabels({'0','\pi/4','\pi/2','3\pi/4','\pi'})
    hold off

    xlabel('|\omega|')
    axis tight
    export_fig(sprintf('curves/radialps_modelA_%s.pdf',name),'-pdf','-transparent')    
end
