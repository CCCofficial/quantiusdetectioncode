%% display expert annotation: one image per click

clc, close all, clear all;

Afull = csvread('../annotation/ResultsAJH_edit_skb.csv');

% image read
prefix = '../data/9_merged';

all_im_ids = [1, 10, 30, 40, 48];

for n = 1 : length(all_im_ids)
    imid = all_im_ids(n);
    x = Afull(Afull(:, 3) == n, 1);
    y = Afull(Afull(:, 3) == n, 2);
    filenm = [prefix num2str(imid, '%0.2d') '.png'];
    im = imread(fullfile('..', filenm));
    figure(10), imagesc(im), colormap gray, axis image, axis on; hold on;
    plot(x, y, 'r*'); title(['Frame no: ', num2str(n, '%0.2d')]); hold off;
    pause;
end