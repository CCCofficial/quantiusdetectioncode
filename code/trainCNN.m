function trainCNN(imdb, iFolds, results_dir, prevmodel, mthd)

if isempty(prevmodel)
    nCh = size(imdb.images.data, 3);
    switch mthd
        case 'smallCNN'
            vggcellnet = initvggSmallCNN(nCh);
        case 'largeCNN'
            vggcellnet = initvggLargeCNN(nCh);
        otherwise 
            error('CNN type mismatch');
    end
    vggcellnet = addCustomLossLayer(vggcellnet, @l2LossForward, @l2LossBackward);
    trainOpts = getopts();
    trainOpts.expDir = results_dir;
else
    trainOpts = getopts();
    vggcellnet = loadvggcellnet(prevmodel, trainOpts.numEpochs, 0, mthd);
    trainOpts.expDir = results_dir;
    trainOpts.batchSize = 2 ;
    trainOpts.numEpochs = 15 ;
    trainOpts.continue = true ;
    trainOpts.learningRate = 10.^(-linspace(4,7,15)); 
end
% data normalization
imdb.images.data = nomalizeIMDB(imdb.images.data, imdb.ch_mean, imdb.ch_std);
% optimization
learnCNN(vggcellnet, trainOpts, imdb, iFolds, mthd);

end

function data = nomalizeIMDB(data, ch_mean, ch_std)
nCh = size(data, 3);
for eachEchannel = 1 : nCh
    data(:,:,eachEchannel,:) = (data(:,:,eachEchannel,:) - ch_mean(eachEchannel))/ch_std(eachEchannel);
end
end