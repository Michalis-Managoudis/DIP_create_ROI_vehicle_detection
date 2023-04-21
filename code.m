clear all;
close all;
clc;

vid = VideoReader('april21.avi'); % read video 
frames = read(vid); % read frames from video

x = vid.Width;
y = vid.Height;
N = vid.NumFrames;
% frames_grayscale = zeros(y,x,N);

nf=250; % frame to display results

figure;
imshow(frames(:,:,:,nf)); % show rgb frame
title('Frame rgb'); axis on;

Gsx=sobel_filter(frames);
%--------------------------------------------------------------------------
%show_excample(Gsx);
%-------------------------------------------------------------------------------
tic;
[masks,lines_] = mask_func(Gsx); % find masks via method 1
masked_frames = apply_mask(frames,masks); % apply masks via method 1
time1=toc;

% figure;
% imshow(frames_mask(:,:,1)*255);
figure;
imshow(masked_frames(:,:,:,nf));

save(masked_frames,'video_masked_1'); % save masked video via method 1 

tic;
[masks2,lines_2] = mask_func_2(Gsx,0.8); % find masks via method 2
masked_frames2 = apply_mask(frames,masks2); % apply masks via method 2
time2=toc;

figure;
imshow(masked_frames2(:,:,:,nf));

save(masked_frames2,'video_masked_2'); % save masked video via method 2
%-----------
% 2
%-----------
frames_noised = add_salt_pepper_noise(add_gaussian_noise(frames)); % add noise to video
save(frames_noised,'video_noised');

Gsx_noise=sobel_filter(frames_noised);

[masks_noise,lines_noise] = mask_func(Gsx_noise); % find masks via method 1 with noise
masked_frames_noise = apply_mask(frames_noised,masks_noise); % apply masks via method 1 with noise

[masks_noise2,lines_noise2] = mask_func_2(Gsx_noise,0.8); % find masks via method 1 with noise
masked_frames_noise2 = apply_mask(frames_noised,masks_noise2); % apply masks via method 1 with noise

figure;
imshow(masked_frames_noise(:,:,:,nf));

figure;
imshow(masked_frames_noise2(:,:,:,nf));

%-------------------
% -----denoising----

frames_denoised = denoise(frames_noised);

figure;
imshow(frames_denoised(:,:,:,nf));

save(frames_denoised,'video_denoised');

% % vid2 = VideoReader('video_denoised.avi'); % read video 
% % frames_denoised = read(vid2); % read frames from video

Gsx_denoise=sobel_filter(frames_denoised);

[masks_denoise,lines_denoise] = mask_func(Gsx_denoise); % find masks via method 1 with noise
masked_frames_denoise = apply_mask(frames_denoised,masks_denoise); % apply masks via method 1 with noise

[masks_denoise2,lines_denoise2] = mask_func_2(Gsx_denoise,0.8); % find masks via method 1 with noise
masked_frames_denoise2 = apply_mask(frames_denoised,masks_denoise2); % apply masks via method 1 with noise

figure;
imshow(masked_frames_denoise(:,:,:,nf));

figure;
imshow(masked_frames_denoise2(:,:,:,nf));

% -----------------------
% 3
% -----------------------

frames_edged = canny_(frames_denoised,lines_denoise2);

figure;
imshow(frames_edged(:,:,nf));



% -------------------------------------------------------------
% functions
% -------------------------------------------------------------
function Gsx = sobel_filter(frames)
    [y,x,~,N]=size(frames);
    for f=1:N
        frames_grayscale(:,:,f) = rgb2gray(frames(:,:,:,f)); % convert frames to grayscale
    end

    % Sobel matrix in x axis 
    Sx=[-1,-2,-1;
         0, 0, 0;
         1, 2, 1];

    for f=1:N
        s1 = conv2(frames_grayscale(:,:,f),Sx); % horizontal sobel filtering
        s = s1(2:y+1,2:x+1); % erase border values
        Gsx(:,:,f) = abs(( (s - min(min(s))) / (max(max(s)) - min(min(s))) -0.5 ) * 2 ); % normalization
    end
end
function show_excample(Gx)
    [y,~,N] = size(Gx);
    for f=1:N
        for line = 1 : y
            sm(line) = sum(Gx(line,:,f)); % sum elements of each line
        end
        hist(:,f) = sm; % make a horizontal histogram
    end

    [max_value,max_index] = max(hist(2:y-1,1));

    figure;
    imshow(Gx(:,:,1),[]); % show sobel filtered frame
    yline(max_index,'r-', 'LineWidth', 2);
    title('Horizontal sobel filtered Frame 1'); axis on;

    figure;
    plot(hist(2:y-1,1)); % show histogram
    xline(max_index,'r--');
    yline(max_value,'r--');
    title('Line-histogramm in Frame 1'); axis on;
end

function [masks,lines] = mask_func(Gx)
    [y,x,N] = size(Gx);
    lines = zeros(1,N);
    masks = zeros(y,x,N);
    
    for f = 1:N
        tmp_hist = zeros(1,y);
        for line=1:y
            tmp_hist(line) = sum(Gx(line,:,f));
        end
        [~,max_index] = max(tmp_hist(2:y-1));
        
        lines(f) = max_index;
        mask_0 = zeros(x,max_index-1);
        mask_1 = ones(x,y - max_index + 1);
        mask = cat(2,mask_0,mask_1)';
        masks(:,:,f) = mask; 
    end
    masks = uint8(masks);
end
function masked_frames = apply_mask(frames,mask_arr)
    [y,x,~,N] = size(frames);
    masked_frames = zeros(y,x,3,N);
    for f=1:N
        new_frame(:,:,1) = frames(:,:,1,f).*mask_arr(:,:,f);
        new_frame(:,:,2) = frames(:,:,2,f).*mask_arr(:,:,f);
        new_frame(:,:,3) = frames(:,:,3,f).*mask_arr(:,:,f);
        masked_frames(:,:,:,f) = new_frame;
    end
    masked_frames = uint8(masked_frames);
end
function save(frames,filename)
    sz = size(frames);
    N = sz(4);
    v = VideoWriter(append(filename,'.avi'),'Uncompressed AVI');
    open(v);
    for f=1:N
        writeVideo(v,frames(:,:,:,f));
    end
    close(v);
end
function [masks,lines] = mask_func_2(Gx,th)
    [y,x,N] = size(Gx);
    lines = zeros(1,N);
    masks = zeros(y,x,N);
    
    for f = 1:N
        tmp_hist = zeros(1,y);
        tmp_hist_smoothed = zeros(1,y);
        for line=1:y
            tmp_hist(line) = sum(Gx(line,:,f));
        end
        for l=2:y-1
            tmp_hist_smoothed(l) = (tmp_hist(l-1) + tmp_hist(l) + tmp_hist(l+1))/3;
        end
        tmp_hist_smoothed(1) = (tmp_hist(l) + tmp_hist(l+1))/3;
        tmp_hist_smoothed(y) = (tmp_hist(l) + tmp_hist(l-1))/3;
        
        tmp_hist_smoothed = tmp_hist_smoothed(2:y-1);        
        
        mx = max(tmp_hist_smoothed(5:y-4));
        
        if (mod(f,10)==1)
            l=5;
            h=y-4;
        else
            l = max(lines(f-1)-10,5);
            h = min(lines(f-1)+10,y-4);
        end
        
        i=l;
        while i<h
            i=i+1;
            if (f==19)
                xxx=5;
            end
            if (tmp_hist_smoothed(i)<th*mx)
                if (i==h)
                    l=5;
                    h=y-4;
                    i=l;
                end
                continue;
            else
                lines(f) = i;
                mask_0 = zeros(x,i-1);
                mask_1 = ones(x,y - i + 1);
                mask = cat(2,mask_0,mask_1)';
                masks(:,:,f) = mask;
                break;
            end
        end 
    end
    masks = uint8(masks);
end

function sig = add_gaussian_noise(imgs)
    sz=size(imgs);
    N=sz(4);
    %sig = zeros(sz);
    for f=1:N
        sig(:,:,:,f)=imnoise(imgs(:,:,:,f),'gaussian');
    end
    %sig = uint8(sig);
end
function sig = add_salt_pepper_noise(imgs)
    sz=size(imgs);
    N=sz(4);
    %sig = zeros(sz);
    for f=1:N
        sig(:,:,:,f)=imnoise(imgs(:,:,:,f),'salt & pepper',0.1);
    end
    %sig = uint8(sig);
end
function res=mov_avg_filter_w_padding(sig,szz)
    sz=size(sig);
    res=zeros(sz);
    pad=floor(szz/2);
    ar=padarray(sig,[pad,pad],128,'both');
    for i=1:sz(1)
        for j=1:sz(2)
            res(i,j)=mean(ar(i:i+2*pad,j:j+2*pad),'all');
        end
    end    
end
function outpt = denoise(inpt)
    sz=size(inpt);
    N=sz(4);
    for f=1:N
        for c=1:3
            outpt(:,:,c,f) = mov_avg_filter_w_padding(medfilt2(inpt(:,:,c,f),[3,3]),3);
        end
    end
    outpt = uint8(outpt);
end

function edges_ = canny_(frm,lines)
    [y,x,~,N]=size(frm);
    edges_ = logical(zeros(y,x,N));
    for f=1:N
        gr_frm(:,:,f) = rgb2gray(frm(:,:,:,f)); % convert frames to grayscale
        frame_edge = edge(gr_frm(lines(f):y,:,f),'Canny',[0.4,0.6]); % apply canny edge detector
        [yy,~] = size(frame_edge);
        edges_(y-yy+1:y,:,f) = frame_edge;
    end
end


