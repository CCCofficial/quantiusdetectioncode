function vggcellnet = loadvggcellnet(expfolder, totalEpoch, kthfold, mthd)
modelPath = @(ep, kfold, method) fullfile(expfolder, sprintf('net-epoch-%d-fold-%d-%s.mat', ep, kfold, method));

load(modelPath(totalEpoch, kthfold, mthd), 'info');

[~, bestModel] = min(info.train.objective);

load(modelPath(bestModel, kthfold, mthd), 'vggcellnet');
