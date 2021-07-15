% Thomas Engels <thomas.engels@mail.tu-berlin.de>

%

function [vor_time,enstrophy,time_list] = OMEGA_1_nopen_ns_randn4_N512(varargin)

close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global params

params.nx           = 512;

params.ny           = 512;

params.Lx           = 2*pi;

params.Ly           = 2*pi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params.nu           = 1e-4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params.CFL          = 0.1;

params.T_end        = 8;

params.iplot        = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% grid

params.y            = params.Ly*(0:params.ny-1)/(params.ny); 

params.x            = params.Lx*(0:params.nx-1)/(params.nx);

params.dx           = params.x(2)-params.x(1);

params.dy           = params.y(2)-params.y(1);

[params.X,params.Y] = meshgrid(params.x,params.y);

params.X            = params.X';

params.Y            = params.Y';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create wavenumber matrices (global)

params.kx             = (2*pi/params.Lx)*[0:(params.nx/2-1) (-params.nx/2):(-1)]; % Vector of wavenumbers

params.ky             = (2*pi/params.Ly)*[0:(params.ny/2-1) (-params.ny/2):(-1)]; % Vector of wavenumbers

[params.Kx,params.Ky] = meshgrid(params.kx,params.ky);

params.Kx             = params.Kx';

params.Ky             = params.Ky';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = params.X;

Y = params.Y;



% ial condition

if nargin == 0
    vork_old = Inicond;
else
    vork_old = fft2(varargin{1});
end

time = 0;

it   = 1;

% filename = 'NS_Video';
% vidObj = VideoWriter(filename,'MPEG-4');
% vidObj = VideoWriter(filename);
% vidObj.FrameRate = 15;
% open(vidObj);
% 
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
% 
% clf


    
vor = cofitxy(vork_old);

% plot(time_list,enstrophy);
% xlabel('time','FontSize',24)
% ylabel('enstrophy','FontSize',24)
% title('Dissiplation','FontSize',24)


vor_time(:,:,1) = vor;
enstrophy(1) = sum(vor(:).^2);
time_list(1) = time;

pcolor(X,Y,vor);

% colormap(PaletteMarieAll('Vorticity',600,0.3,5,0.25));

scale = 1.0;

axis equal

c = scale*max ( min(min(abs(vor))), max(max(vor)) );

caxis([-c c])

colorbar

shading interp

title(['Navier-Stokes Vorticity, t=' num2str(time) ' dt=n/a' ])

drawnow  

drawpsd(vor);

% currFrame = getframe(fig);
% writeVideo(vidObj,currFrame);

while (time<params.T_end)

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   [vork_new, params.dt] = RK2(vork_old);  

   vork_old = vork_new;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



   time = time + params.dt;

   it = it+1;



   if (mod(it,params.iplot)==0) || time>params.T_end      

    figure(1)
    clf

    vor = cofitxy(vork_new);
    if (mod(it,params.iplot)==0)
    vor_time(:,:,(it/params.iplot)+1) = vor;
    enstrophy((it/params.iplot)+1) = sum(vor(:).^2);
    time_list((it/params.iplot)+1) = time;
    else
    vor_time(:,:,end+1) = vor;
    enstrophy(end+1) = sum(vor(:).^2);
    time_list(end+1) = time;
    end

    pcolor(X,Y,vor);

    colormap(PaletteMarieAll('Vorticity',600,0.3,5,0.25));
%     colormap gray
    
    scale = 1.0;

    axis equal

    c = scale*max ( min(min(abs(vor))), max(max(vor)) );

    caxis([-c c])

    colorbar
    
    shading interp

    title(['Navier-Stokes Vorticity, t=' num2str(time) ' dt=' num2str(params.dt,'%e') ])

    drawnow    
   
    drawpsd(vor);


    % currFrame = getframe(fig);
    % writeVideo(vidObj,currFrame);

   end 

end

drawpsd(vor)

end




%..........................................
function drawpsd(vor)

figure(2)
[vorS,vorK]=Spectre2D(vor);
% fit with C*k^alpha
alpha=-2;
C=exp(max(log(vorS)-alpha*log(vorK)));
% subplot(2,1,1)
plot(vorK,vorS);
hold on
plot(vorK,C*(vorK.^alpha))
hold off
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
xlabel('k','FontSize',24)
ylabel('Z(k)','FontSize',24)
title('Enstropy sepctra','FontSize',24)
%ylim([vorS(104),max(vorS)])
%xlim([20,110])
%pause
end

function [S,K]=Spectre2D(A)

siz=size(A);
[y,x] = meshgrid(1:siz(2),1:siz(1));
x=x-(siz(1)/2+1);
y=y-(siz(2)/2+1);
modx=fftshift(sqrt(x.^2 + y.^2));

K=1:(min(siz(1),siz(2))/2);
S=0.*K;

E = abs(fft2(A)).^2;

for i=K
 mask = ((modx>=i)&(modx<(i+1)));
 S(i) = sum(sum(E.*mask));
end

end


function [nlk, dt]=nonlinear2(vork)

    % computes the non-linear terms + penalization in Fourier space.    

    global params

    uk = Velocity(vork);

    u(:,:,1) = cofitxy(uk(:,:,1));

    u(:,:,2) = cofitxy(uk(:,:,2));

    

    % deterimne time step 

    dt = params.CFL*params.dx/max(max(max(abs(u))));      



    nlk = fft2( -u(:,:,1).*cofitxy(1i*params.Kx.*vork) - u(:,:,2).*cofitxy(1i*params.Ky.*vork)  );

end







function [vork_new,dt] = RK2(vork)

    global params

  

    % compute non-linear terms

    [nlk,dt] = nonlinear2(vork);



    % advance in time

    vis = exp(-params.nu*dt*(params.Kx.^2+params.Ky.^2) );

    

    vork_new = vis.*(vork + dt*nlk );                

    
    % Dealiasing (Kai suggested I add this extra dealiasing in)

    vork_new = Dealias(vork_new);
    
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % END OF EULER STEP

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

    [nlk2,dt_dummy] = nonlinear2(vork_new);

     

    % advance in time

    vork_new = vis.*vork + 0.5*dt*(nlk.*vis + nlk2);        

  

    % Dealiasing

    vork_new = Dealias(vork_new);

end





function vork = Inicond()

    global params

    vor  = 10*randn(params.nx,params.ny); % 100*(2*rand(params.nx,params.ny)-1); 
    gab = gabor_2d_period(params.nx,params.ny, 0.8, 1, 0, 0, [0,0], 'double', 4); % gabor J=1
    
%     imagesc(log10(abs(fftshift(fft2(gab)))))
%     drawpsd(gab)
    
    vork = fft2(gab).* fft2(vor); % smooth it
    % vork = exp(-0.01*(params.Kx.^2+params.Ky.^2) ) .* fft2(vor); % smooth it
end


function uk = Poisson(Sk)

    global params

    K = -(params.Kx.^2 + params.Ky.^2); % laplacian is division -|k|^2

    uk = zeros(size(Sk));

    uk(K<0) = Sk(K<0)./K(K<0);    

end











function d = Divergence(ukx,uky)

    global params

    d = 1i*params.Kx.*ukx + 1i*params.Ky.*uky;

end





function vork = Vorticity(ukx,uky)

    global params

    vork = 1i*params.Kx.*uky  - 1i*params.Ky.*ukx;

end





function uk = Velocity(vork)

    global params

    stream = Poisson(vork);

    uk(:,:,1) = -1i*params.Ky.*stream;

    uk(:,:,2) = +1i*params.Kx.*stream;

end







function uk = Dealias(uk)

    global params

    K=(params.Kx.^2+params.Ky.^2);

    kcut = (1/3)*max(max(K));    

    uk(K>kcut) =0;

end





function f = cofitxy(fk)

    f = ifft2(fk,'symmetric');

end





function PaletteMarie = PaletteMarieAll(Type,taille,limite_faible_fort,etalement_du_zero,blackmargin)



    % PaletteMarie(Type,taille,limite_faible_fort,etalement_du_zero)

    % Where Type is : 'Vorticity'



    % Modified after discussion with Marie 14/02/2010

    % Modified after discussion with Marie  3/11/2010



    if strcmp(Type,'Pressure')

         color1=[0 0.5 1];     

         color2=[1 1 1];

         color3=[1 1 0];

         zero=[1 0 0];

         limite_basse = floor(taille/2*(1-limite_faible_fort));

         limite_haute = ceil(taille/2*(1+limite_faible_fort));

         zero_moins = floor((taille-etalement_du_zero)/2);

         zero_plus = ceil((taille + etalement_du_zero)/2);

        PaletteMarie = [ linspace(blackmargin,1,limite_basse)'*color1; linspace(blackmargin^3,0.5,zero_moins-limite_basse)'*color2; ones(etalement_du_zero,1)*zero; linspace(0.5,1-blackmargin^3,limite_haute-zero_plus)'*color2; linspace(blackmargin,1,taille-limite_haute)'*color3 ];

    end 





    if strcmp(Type,'Streamfunction') % Also recommended for the velocity (3 nov 2010)

         color1=[0.5 0 1];

         color2=[1 1 1];

         color3=[1 0.8 0];

         zero=[0.0 1.0 0.5];

         limite_basse = floor(taille/2*(1-limite_faible_fort));

         limite_haute = ceil(taille/2*(1+limite_faible_fort));

         zero_moins = floor((taille-etalement_du_zero)/2);

         zero_plus = ceil((taille + etalement_du_zero)/2);

    PaletteMarie = [ linspace(blackmargin,1,limite_basse)'*color1; linspace(blackmargin^3,0.5,zero_moins-limite_basse)'*color2; ones(etalement_du_zero,1)*zero; linspace(0.5,1-blackmargin^3,limite_haute-zero_plus)'*color2; linspace(blackmargin,1,taille-limite_haute)'*color3 ];

    end 





    if strcmp(Type,'Velocity')

         color1=[1 1 0];

         color2=[1 1 1];

         color3=[1 0.5 0.5];

         zero=[0.5 1 0.5];

         limite_basse = floor(taille/2*(1-limite_faible_fort));

         limite_haute = ceil(taille/2*(1+limite_faible_fort));

         zero_moins = floor((taille-etalement_du_zero)/2);

         zero_plus = ceil((taille + etalement_du_zero)/2);

    PaletteMarie = [ linspace(blackmargin,1,limite_basse)'*color1; linspace(blackmargin^3,0.5,zero_moins-limite_basse)'*color2; ones(etalement_du_zero,1)*zero; linspace(0.5,1-blackmargin^3,limite_haute-zero_plus)'*color2; linspace(blackmargin,1,taille-limite_haute)'*color3 ];

    end 





    if strcmp(Type,'Vorticity')

         color1=[0 0.5 1];     %light blue

         color2=[1 1 1];       %white

         color3=[1 0.5 0.5];   %light red

         zero=[255 222 17]/255;

         limite_basse = floor(taille/2*(1-limite_faible_fort));

         limite_haute = ceil(taille/2*(1+limite_faible_fort));

         zero_moins = floor((taille-etalement_du_zero)/2);

         zero_plus = ceil((taille + etalement_du_zero)/2);



    PaletteMarie = [ linspace(blackmargin,1,limite_basse)'*color1;...

                     linspace(blackmargin^3,0.5,zero_moins-limite_basse)'*color2;...

                     ones(etalement_du_zero,1)*zero;...

                     linspace(0.5,1-blackmargin^3,limite_haute-zero_plus)'*color2;...

                     linspace(blackmargin,1,taille-limite_haute)'*color3 ];

    end

end



