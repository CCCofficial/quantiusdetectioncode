function finetuneCNN(imdb, results_dir, prevmodel, mthd)
trainOpts = getopts();
vggcellnet = loadvggcellnet(prevmodel, trainOpts.numEpochs, 0, mthd);
trainOpts.expDir = results_dir;
trainOpts.batchSize = 2 ;
trainOpts.numEpochs = 15 ;
trainOpts.continue = true ;
trainOpts.learningRate = 10.^(-linspace(2,7,15));

% data normalization
imdb.images.data = nomalizeIMDB(imdb.images.data, imdb.ch_mean, imdb.ch_std);
% optimization
learnCNN(vggcellnet, trainOpts, imdb, 0, mthd);

end

function data = nomalizeIMDB(data, ch_mean, ch_std)
nCh = size(data, 3);
for eachEchannel = 1 : nCh
    data(:,:,eachEchannel,:) = (data(:,:,eachEchannel,:) - ch_mean(eachEchannel))/ch_std(eachEchannel);
end
end