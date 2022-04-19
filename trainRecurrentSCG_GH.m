function trainRecurrentSCG_GH(varargin)

rng('shuffle');

global Li;
global Lr;
global Lo;

%%% Size of network
N = 30;
trainCG        = 1;


trainNet       = 1;
prune          = 0;  %%% 1 = prune, -1 = load prune, 0 = nothing
probeTraj      = 0;
trainPrune     = 0;

%%% 2 == retrain
trialType      = 1;
plotTrainPrune = 0;


% xDim.weightsigma     = 0.00;
% xDim.sigmay          = 0.1;
% xDim.sigmax          = 0;
% xDim.trialRepeats    = 4;
% xDim.trainingRepeats = 5;
% 
% prunevs = [0.1 : 0.05 : 0.95];

xDim.weightsigma     = 0.00; %%% perturb weights of unpruned network
xDim.sigmay          = 0.1;
xDim.sigmax          = 0;
xDim.trialRepeats    = 4;
xDim.trainingRepeats = 5;

prunevs = [0.1 : 0.05 : 0.95];


Li = 5+1;
Lr = N;
Lo = 1;
    

paralleli = -1;
trainPruneID = 1; %%% 1 == prunednetwork, 2 = unpruned

    
if trainNet == 1
    
    Wi = 0.2*randn(Lr, Li);
    Wr = 0.1*randn(Lr, Lr) + 0.5*eye(Lr, Lr);
    Wo = 0.2*randn(Lo, Lr);
        
    fixedIter = -1;
    
    for trainingIteraction = 1 : xDim.trainingRepeats
                
        [X, Y, xDim] = buildTrials(trialType, xDim);

        [Wi, Wr, Wo, ssefo, iter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim);
        
    end

    weightName = sprintf('weightMatSCG_%d_%d_%d.mat', N, trialType, paralleli);
    
    save(weightName, 'Wi', 'Wr', 'Wo', 'ssefo', 'iter', 'xDim');
    
elseif plotTrainPrune ~= 1
        
%     weightName = sprintf('weightMatSCG_%d_%d_%d.mat', N, trialType, paralleli);
%     load(weightName);
    
end


if trialType == 1

    if prune == 1
        
        if trainPruneID == 1 || paralleli == -1
        
            %%% Pruned network
            for prunei = 1 : length(prunevs)

                prunev = prunevs(prunei);

                if prunei > 1                

                    prunevl = prunevs(prunei-1);

                    weightName = sprintf('prunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunevl, paralleli);
                    load(weightName);

                    Wi = Wipruned;
                    Wr = Wrpruned;
                    Wo = Wopruned;

                end

                [Wrpruned, clampVec] = pruneNet(Wi, Wr, Wo, prunev);

                fixedIter = -1;

                [X, Y, xDim] = buildTrials(trialType, xDim);

                [Wipruned, Wrpruned, Wopruned, ssefo, iter] = trainNetwork(trainCG, Wi, Wrpruned, Wo, X, Y, trialType, fixedIter, xDim, clampVec);

                weightName = sprintf('prunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

                save(weightName, 'Wipruned', 'Wrpruned', 'Wopruned', 'ssefo', 'iter', 'xDim');

            end
        end
        
        if trainPruneID == 2 || paralleli == -1
        
            %%% continue training unpruned network with perturbations
            for prunei = 1 : length(prunevs)

                prunev = prunevs(prunei);

                if prunei > 1                

                    prunevl = prunevs(prunei-1);

                    weightName = sprintf('unprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunevl, paralleli);
                    load(weightName);

                end
                
                Wr = Wr + xDim.weightsigma*randn(size(Wr, 1), size(Wr, 2));

                fixedIter = -1;

                [X, Y, xDim] = buildTrials(trialType, xDim);

                [Wi, Wr, Wo, ssefo, iter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim);

                weightName = sprintf('unprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

                save(weightName, 'Wi', 'Wr', 'Wo', 'ssefo', 'iter', 'xDim');

            end      
        end
        

    elseif prune == -1
        
        runSummary = 1;
        
        nFiles = 400;
        
        mptrial  = NaN(nFiles, length(prunevs), 1, 6);
        mntrial  = NaN(nFiles, length(prunevs), 1, 6);
        dp       = NaN(nFiles, length(prunevs), Lr);
        dn       = NaN(nFiles, length(prunevs), Lr);
        
        pvar     = NaN(nFiles, length(prunevs), 6, 4);
        lyapunov = NaN(nFiles, length(prunevs), 2);
        
        crct     = NaN(nFiles, length(prunevs), 6, 6, 2);
        err      = NaN(nFiles, length(prunevs), 6, 6, 2);
        
        ssef_f = NaN(nFiles, length(prunevs), 2, 50);
        
        mnsse  = NaN(nFiles, length(prunevs), 2);
        miter  = NaN(nFiles, length(prunevs), 2);
        
        pruneHist = [-1.5 : 0.05 : 1.5];
        
        cDist  = NaN(nFiles, length(prunevs), 2, length(pruneHist));
        
        for prunei = 1 : length(prunevs)
           prunev = prunevs(prunei);
           legendTitle{prunei} = sprintf('%.2f', prunev);
        end
        
        nPrune = length(prunevs);
        
        if runSummary == 0
            for filei = ffile : (ffile-1)+nFiles

                fprintf('File %d\n', filei);

                for prunei = 1 : nPrune

                    prunev = prunevs(prunei);
                    pweightName = sprintf('prunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, filei);
                    uweightName = sprintf('unprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, filei);

                    [Xtrial, Ytrial, xDim] = buildTrials(trialType, xDim);

                    try 
                        load(pweightName);
                        ssefop = ssefo;
                        load(uweightName);
                        ssefou = ssefo;

                        ssef_f(filei, prunei, 1, 1:length(ssefou)) = ssefou;
                        ssef_f(filei, prunei, 2, 1:length(ssefop)) = ssefop;

                        zp = isnan(ssefop);
                        zu = isnan(ssefou);

                        zp = find(zp == 0);
                        zp = zp(end);
                        zu = find(zu == 0);
                        zu = zu(end);

                        mnsse(filei-ffile+1, prunei, 1) = ssefou(zu);
                        mnsse(filei-ffile+1, prunei, 2) = ssefop(zp);
                        miter(filei-ffile+1, prunei, 1) = zu;
                        miter(filei-ffile+1, prunei, 2) = zp;

                        [flyapunovu, flyapunovp, mcrct, merr, dimp, dimu] = ...
                            outputPredictions(Wi, Wr, Wo, Wipruned, Wrpruned, Wopruned, ssefou, ssefop, xDim);
                        
                        [Nn, Np] = connectivityDistribution(Wr, Wrpruned);

                        cDist(filei-ffile+1, prunei, 1, :) = Nn;
                        cDist(filei-ffile+1, prunei, 2, :) = Np;                    

                        if flyapunovu > -Inf
                            lyapunov(filei-ffile+1, prunei, 1) = flyapunovu;
                        end

                        if flyapunovp > -Inf
                            lyapunov(filei-ffile+1, prunei, 2) = flyapunovp;
                        end

                        crct(filei-ffile+1, prunei, :, :, :) = mcrct;
                        err(filei-ffile+1, prunei, :, :, :)  = merr;

                        dp(filei-ffile+1, prunei, :) = dimp/sum(dimp);
                        dn(filei-ffile+1, prunei, :) = dimu/sum(dimu);

                        fprintf('');
                    catch
                        fprintf('Missing file %d %.2f\n', filei, prunev);
                    end

                end
            end
            
        else
            
            load('AllRecurrentSCG.mat');
            
        end
            
        
        mlyapunov = squeeze(nanmean((lyapunov), 1));
        slyapunov = squeeze(nanstd((lyapunov), 1, 1))/sqrt(nFiles);
        
        %%% note this should be 13 for 0.7 
        pbin = 14;
        
        figure
        
        subplot(2,2,1);
        errorbar([prunevs' prunevs'], squeeze(nanmean(mnsse, 1)), squeeze(nanstd(mnsse, 1, 1))/sqrt(nFiles));
        
        
        subplot(2,2,2);
        errorbar([prunevs' prunevs'], squeeze(nanmean(miter, 1)), squeeze(nanstd(miter, 1, 1))/sqrt(nFiles));
        
        
        subplot(2,2,3);
        tplotu = squeeze(nanmean(ssef_f(:, :, 1, 1:30), 1));
        tplotp = squeeze(nanmean(ssef_f(:, :, 2, 1:30), 1));
        
        vtplotu = reshape(tplotu', 1, prod(size(tplotu)));
        vtplotp = reshape(tplotp', 1, prod(size(tplotu)));
        
        mssef = squeeze(mean(nanmean(ssef_f(:, 1:7, :, 1:30), 1), 2))';
        plot(1:length(vtplotu), vtplotu, 1:length(vtplotp), vtplotp);

        subplot(2,2,4);
        [t, p, ci, stats] = ttest(squeeze(crct(:, :, :, :, 2) - crct(:, :, :, :, 1)));
        surf(squeeze(stats.tstat(1, 11, :, :)));
        view(0, 90);
        colorbar;
        
        dfcrct = squeeze(crct(:, :, 1, 1, 2) - crct(:, :, 1, 1, 1));
        
        histu = squeeze(nanmean(cDist(:, pbin, 1, :), 1));
        histp = squeeze(nanmean(cDist(:, pbin, 2, :), 1));
        
        plot(pruneHist, histu/sum(histu), pruneHist, histp/sum(histp));

        
        figure;       
        subplot(2,2,1);
        errorbar([prunevs' prunevs'], squeeze(mean(nanmean(nanmean(crct(:, :, 1:6, 2:end, :), 1), 3), 4)), ...
                                          squeeze(mean(mean(nanstd(crct(:, :, 1:6, 2:end, :), 1, 1), 3), 4))/sqrt(nFiles));
        
        [p1, pp1]  = ttest(squeeze(mean(nanmean(crct(:, :, 1:6, 2:end, 1), 3), 4)) - ...
                           squeeze(mean(nanmean(crct(:, :, 1:6, 2:end, 2), 3), 4)));
        
        subplot(2,2,2);
        errorbar(squeeze(mean(nanmean(crct(:, pbin, :, 2:end, :), 1), 4)), ...
                 squeeze(mean(nanstd(crct(:, pbin, :, 2:end, :), 1, 1), 4))/sqrt(nFiles));
        title('Time')
        
        [p2, pp2] = ttest(squeeze(mean(crct(:, pbin, :, 2:end, 1), 4)) - ...
                          squeeze(mean(crct(:, pbin, :, 2:end, 2), 4)));
        
        subplot(2,2,3);
        errorbar(squeeze(mean(nanmean(crct(:, pbin, :, :, :), 1), 3)), ...
                 squeeze(mean(nanstd(crct(:, pbin, :, :, :), 1, 1), 3))/sqrt(nFiles));
        title('Strength')
        
        [p3, pp3] = ttest(squeeze(mean(crct(:, pbin, :, :, 1), 3)) - ...
                          squeeze(mean(crct(:, pbin, :, :, 2), 3)));
               
        subplot(2,2,4);
        errorbar([prunevs' prunevs'], mlyapunov, slyapunov);
        legend('Unpruned', 'Pruned');
        
        [p4, pp4] = ttest(lyapunov(:, :, 1) - lyapunov(:, :, 2));

        figure;
        %%% unpruned
        crctbinu = squeeze(crct(:, pbin, 2, 2, 1));
        lyapbinu = squeeze((lyapunov(:, pbin, 1)));

        %%% pruned
        crctbinp = squeeze(crct(:, pbin, 2, 2, 2));
        lyapbinp = squeeze((lyapunov(:, pbin, 2)));
        
        [~, zui] = sort(lyapunov(:, pbin, 1));
        [~, zpi] = sort(lyapunov(:, pbin, 2));
        
        pcCtr = 1;
        epsi = 0.1;
        
        for pci = 0.25 : 0.25 : 1
            
            dtip = find(crctbinp > pci-epsi & crctbinp < pci+epsi);
            dtiu = find(crctbinu > pci-epsi & crctbinu < pci+epsi);
            mnp(pcCtr) = nanmean(lyapbinp(dtip));
            snp(pcCtr) = nanstd(lyapbinp(dtip))/sqrt(length(dtip));
            
            mnu(pcCtr) = nanmean(lyapbinu(dtiu));
            snu(pcCtr) = nanstd(lyapbinu(dtiu))/sqrt(length(dtiu));
            
            pcCtr = pcCtr + 1;
            
        end
        
        
        subplot(2,2,1);
        
        xv = 0.25 : 0.25 : 1;
        errorbar([xv' xv'], [mnu' mnp'], [snp' snu']);
        legend('Unpruned', 'Pruned');
        
        zp = find(~isnan(lyapbinp));
        zu = find(~isnan(lyapbinu));
                
        [cru, pvu] = corrcoef([crctbinu(zu) lyapbinu(zu)]);
        [crp, pvp] = corrcoef([crctbinp(zp) lyapbinp(zp)]);
        
        title([crp(1,2) cru(1,2)]);
        
        subplot(2,2,2);
        
        cdp = squeeze(cumsum(dp(:, pbin, 1:8), 3));
        cdn = squeeze(cumsum(dn(:, pbin, 1:8), 3));
        
        dps1 = cumsum(dp(zpi(1:200), :, :), 3);
        dps2 = cumsum(dp(zpi(201:400), :, :), 3);

        [zd, zp] = ttest(squeeze(dps1(:, pbin, 1:8) - dps2(:, pbin, 1:8)));
        
        zp
        
        [zd1, zp2] = ttest(cdn - cdp);
        
        zp2
        
        xvi = find(zp(1:8) < 0.01);
        
        plot(1:8, nanmean(cdn, 1), 1:8, nanmean(cdp, 1));
        legend('Unpruned', 'Pruned');
        
        hold on;
        plot(xvi, ones(length(xvi), 1));
        
        dctr = 1;
        for filei = 1 : nFiles
            for prunei = 1 : length(prunevs)
                for bini = 1 : size(pvar, 3)
                    depvar(dctr, 1) = squeeze(pvar(filei, prunei, bini, 3));
                    
                    if isnan(depvar(dctr, 1))
                        continue;
                    end
                    
                    vfile(dctr, 1)  = filei;
                    vprune(dctr, 1) = prunei;
                    vbin(dctr, 1)   = bini;
                    vptyp(dctr, 1)  = 0;                   
                    
                    dctr = dctr + 1;
                    
                    depvar(dctr, 1) = squeeze(pvar(filei, prunei, bini, 4));
                    
                    if isnan(depvar(dctr, 1))
                        continue;
                    end
                    
                    vfile(dctr, 1)  = filei;
                    vprune(dctr, 1) = prunei;
                    vbin(dctr, 1)   = bini;
                    vptyp(dctr, 1)  = 1;                   
                    
                    dctr = dctr + 1;
                    
                end
            end
        end
        
%         [p, tble] = anovan(depvar, {vfile, vprune, vbin, vptyp}, 'random', [1], 'model', 'interaction');
                        
    end
    
    
    if probeTraj == 1
        
        %%% 7, 15, 32 for unpruned
        %%% 54, 70, 73, 84 for unpruned
        
%         for filei = 61 : 400
        
            pweightName = sprintf('prunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, 0.7, 7);
            uweightName = sprintf('unprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, 0.7, 70);

            load(pweightName);

            load(uweightName);

            [Xtrial, Ytrial, xDim] = buildTrials(trialType, xDim);
                        
            [mproju, tcontractu, lyapunovu, correctu] = probeTrajectories(Wi, Wr, Wo, Xtrial, Ytrial, xDim);
            
%             if  (log(max(lyapunovu))/(2*25) < 0.1 || correctu(4) > 0.3)
%                 continue;
%             end  
            
            [mprojp, tcontractp, lyapunovp, correctp] = probeTrajectories(Wipruned, Wrpruned, Wopruned, Xtrial, Ytrial, xDim);
            
%             if  (log(max(lyapunovp))/(2*25) > 0.1 || correctp(4) < 0.7)
%                 continue;
%             end
            
            figure;
            subplot(2,2,1);        
            plot(1:length(mproju), mproju, 1:length(mprojp), mprojp);
            legend('unpruned', 'pruned');
%             axis([16 40 0 0.7]);

            subplot(2,2,2);
            plot(1:length(tcontractu), tcontractu, 1:length(tcontractp), tcontractp);
%             axis([16 40 0 1.3]);

            subplot(2,2,3);
            plot(1:length(lyapunovu), log(abs(lyapunovu))/(2*25), 1:length(lyapunovp), log(abs(lyapunovp))/(2*25));


            fprintf('lambda pruned %.3f unpruned %.3f assuming %d length\n', log(max(lyapunovp))/(2*25), log(max(lyapunovu))/(2*25), 25);

            connectivityDistribution(Wr, Wrpruned);
            
            fprintf('');
            
%         end
        
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% retrain networks on new option
if trialType == 2 && plotTrainPrune == 0
    
    prunei = find(abs(prunevs - 0.7) < 0.01);
    
    fprintf('File %d\n', paralleli);

    prunev = prunevs(prunei);
    pweightName = sprintf('prunedweightMatSCG_%d_%d_%.2f_%d.mat', N, 1, prunev, paralleli);
    uweightName = sprintf('unprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, 1, prunev, paralleli);

    try 

        fixedIter = 1;

        [X, Y, xDim] = buildTrials(trialType, xDim);
        
        if trainPruneID == 1 || paralleli == -1

            load(pweightName);
            
            [Wrpruned, clampVec] = pruneNet(Wipruned, Wrpruned, Wopruned, prunev);

            [Wipruned, Wrpruned, Wopruned, ssefo, iter] = trainNetwork(trainCG, Wipruned, Wrpruned, Wopruned, X, Y, trialType, fixedIter, xDim, clampVec);

            weightName = sprintf('retrainPrunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

            save(weightName, 'Wipruned', 'Wrpruned', 'Wopruned', 'ssefo', 'iter', 'xDim');
            
        end
        
        if trainPruneID == 2 || paralleli == -1

            load(uweightName);
            
            [Wi, Wr, Wo, ssefo, iter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim);

            weightName = sprintf('retrainUnprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

            save(weightName, 'Wi', 'Wr', 'Wo', 'ssefo', 'iter', 'xDim');
            
        end

    catch
        fprintf('Missing file %d %.2f\n', filei, prunev);
    end
end

if trialType == 2 && plotTrainPrune == 1
    
    prunei = find(abs(prunevs - 0.7) < 0.01);
    
    fprintf('File %d\n', paralleli);

    prunev = prunevs(prunei);
    
    nFiles = 100;

    ssefopruned   = NaN(nFiles, 75);
    ssefounpruned = NaN(nFiles, 75);
    
    for paralleli = 1 : 100
        
        pweightName = sprintf('retrainPrunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

        try

            load(pweightName);  
                
            ssefopruned(paralleli, 1:length(ssefo)) = ssefo;
            
        catch
            
            fprintf('Missing pruned %d\n', paralleli);
            
        end
        
        uweightName = sprintf('retrainUnprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);
        
        try
            load(uweightName);    
        
            ssefounpruned(paralleli, 1:length(ssefo)) = ssefo;
            
        catch
            
            fprintf('Missing unpruned %d\n', paralleli);
            
        end
        
%         outputPredictions(Wi, Wr, Wo, Wipruned, Wrpruned, Wopruned, ssefou, ssefop, xDim);
        
    end
    
    mssefop = nanmean(ssefopruned);
    vssefop = nanstd(ssefopruned)/sqrt(nFiles);
    
    mssefou = nanmean(ssefounpruned);
    vssefou = nanstd(ssefounpruned)/sqrt(nFiles);
    
    nIter = 2:length(mssefop);
    
    figure;
    errorbar([nIter' nIter'], [mssefou(2:end)' mssefop(2:end)'], [vssefou(2:end)' vssefop(2:end)']);
%     axis([0 50 0.025 0.06]);

    xlabel('Training iterations');
    ylabel('Loss');
    
    legend('Unpruned', 'Pruned');
    
end

%%% Directly train a pruned network from scratch
if trainPrune == 1
    
    if paralleli == -1
    
        parfor paralleli = 1 : 1

            if trainPruneID == 1

                trainPruned(paralleli, Lr, Li, Lo, trialType, trainCG, N, xDim);

            else

                trainUnPruned(paralleli, Lr, Li, Lo, trialType, trainCG, N, xDim);

            end

        end
        
    else
        
        if trainPruneID == 1

            trainPruned(paralleli, Lr, Li, Lo, trialType, trainCG, N, xDim);

        else

            trainUnPruned(paralleli, Lr, Li, Lo, trialType, trainCG, N, xDim);

        end
        
    end
    
end

%%% SCG train random prune directory.
if plotTrainPrune == 1
        
    ssefopruned   = NaN(200, 100);
    ssefounpruned = NaN(200, 100);
    
    for paralleli = 1 : 200
        
        weightName = sprintf('trainprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, 0.7, paralleli);

        try

            load(weightName);  
                
            ssefopruned(paralleli, 1:length(ssefo)) = ssefo;
            
        catch
            
            fprintf('Missing pruned %d\n', paralleli);
            
        end
        
        weightName = sprintf('trainunprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, 0, paralleli);
        
        try
            load(weightName);    
        
            ssefounpruned(paralleli, 1:length(ssefo)) = ssefo;
            
        catch
            
            fprintf('Missing unpruned %d\n', paralleli);
            
        end
        
    end
    
    mssefop = nanmean(ssefopruned);
    vssefop = nanstd(ssefopruned)/sqrt(200);
    
    mssefou = nanmean(ssefounpruned);
    vssefou = nanstd(ssefounpruned)/sqrt(200);
    
    nIter = 2:length(mssefop);
    
    figure;
    errorbar([nIter' nIter'], [mssefou(2:end)' mssefop(2:end)'], [vssefou(2:end)' vssefop(2:end)']);
%     axis([0 50 0.025 0.06]);

    xlabel('Training iterations');
    ylabel('Loss');
    
    legend('Unpruned', 'Pruned');
       
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lyapunovu, lyapunovp, mcrct, merr, dimp, dimu] = outputPredictions(Wi, Wr, Wo, ...
    Wipruned, Wrpruned, Wopruned, ssefou, ssefop, xDim)

[Xtrial, Ytrial, xDim] = buildTrials(1, xDim);

probeStrength = [0:0.2:1.0];
% probeStrength = 0;

probeBinTimes = 16 : 4 : 36;
% probeBinTimes = 18;

trialProbeValue = [-1 1 -1 1];

T = size(Xtrial, 3);

T = size(Xtrial, 3);
nDims = size(Wr, 1);


nDims = size(Wr, 1);
AUnpruned = zeros(xDim.nConditions, nDims, T);
APruned = zeros(xDim.nConditions, nDims, T);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%PCA axes
Tuse = T-15;

Aftunpruned = NaN(nDims, xDim.nConditions*Tuse);
Aftpruned   = NaN(nDims, xDim.nConditions*Tuse);

crct = zeros(4, length(probeBinTimes), length(probeStrength), 2);
err  = zeros(4, length(probeBinTimes), length(probeStrength), 2);
    
% figure;
for triali = 1 : xDim.nConditions

    vi = rem(triali, 4);
    if vi == 0
        vi = 4;
    end
    
    Xi = squeeze(Xtrial(triali, :, :));

    [yhatu_n, actUnpruned] = forwardPass(Wi, Wr, Wo, Xi, Ytrial(triali, :));
    [yhatp_n, actPruned] = forwardPass(Wipruned, Wrpruned, Wopruned, Xi, Ytrial(triali, :));
    
    AUnpruned(triali, :, :) = actUnpruned;
    APruned(triali, :, :)   = actPruned;

    for bini = 1 : length(probeBinTimes)
            
        binTime = probeBinTimes(bini);
        
        for stri = 1 : length(probeStrength)

            Xi = squeeze(Xtrial(triali, :, :));

            Xi(2, binTime:(binTime+1)) = probeStrength(stri)*trialProbeValue(vi);

            [yhatu_p, aunpruned] = forwardPass(Wi, Wr, Wo, Xi, Ytrial(triali, :));
            [yhatp_p, apruned] = forwardPass(Wipruned, Wrpruned, Wopruned, Xi, Ytrial(triali, :));
            
            if bini == 2 && stri == 3 && triali < 5
                
                sbin = (triali-1)*Tuse + 1;
                ebin = sbin + Tuse - 1;
                Aftunpruned(:, sbin:ebin) = aunpruned(:, 16:end);
                Aftpruned(:, sbin:ebin)   = apruned(:, 16:end);
                
            end

    %         if triali <= 4
    % 
    %             subplot(4,4,triali);
    %             plot(1:T, squeeze(Xtrial(triali,2, :)), 1:T, squeeze(Xtrial(triali, 3, :)), ...
    %                 1:T, yhatu_n, 1:T, squeeze(mean(Ytrial(triali:4:end, :), 1)));
    %             title('Unpruned unprobed');
    % 
    %             subplot(4,4,triali+4);
    %             plot(1:T, squeeze(Xtrial(triali,2, :)), 1:T, squeeze(Xtrial(triali, 3, :)), ...
    %                 1:T, yhatp_n, 1:T, squeeze(mean(Ytrial(triali:4:end, :), 1)));
    %             title('Pruned unprobed');
    % 
    %             subplot(4,4,triali+8);
    %             plot(1:T, squeeze(Xi(2, :)), 1:T, squeeze(Xi(3, :)), ...
    %                 1:T, yhatu_p, 1:T, squeeze(mean(Ytrial(triali:4:end, :), 1)));
    %             title('Unpruned probed');
    % 
    %             subplot(4,4,triali+12);
    %             plot(1:T, squeeze(Xi(2, :)), 1:T, squeeze(Xi(3, :)), ...
    %                 1:T, yhatp_p, 1:T, squeeze(mean(Ytrial(triali:4:end, :), 1)));
    %             title('Pruned probed');
    %             
    %         end


            mnY = squeeze(mean(mean(Ytrial(vi:4:end, 52:53), 1), 2));
            

            mnU = mean(yhatu_p(52:53));
            mnP = mean(yhatp_p(52:53));

            crct(triali, bini, stri, 1) = abs(mnY - mnU) < 1;
            crct(triali, bini, stri, 2) = abs(mnY - mnP) < 1;

            if max(yhatp_p(1:51)) > 0.75
                err(triali, bini, stri, 2) = 1;
            end

            if max(yhatu_p(1:51)) > 0.75
                err(triali, bini, stri, 1) = 1;
            end

            fprintf('');
        end
    end
end


cvApruned   = cov(squeeze(AUnpruned(1, :, :))');
cvAunpruned = cov(squeeze(APruned(1, :, :))');

[~, Dpruned] = eig(cvApruned);
[~, Dunpruned] = eig(cvAunpruned);

dpruned   = diag(Dpruned);
dunpruned = diag(Dunpruned);

dimp = dpruned(end:-1:1);
dimu = dunpruned(end:-1:1);

crct = squeeze(mean(crct, 5));
err = squeeze(mean(err, 5));


tOn  = probeBinTimes(1);
tOff = probeBinTimes(end);

conditioni = 4;

for ti = tOn : tOff
    
    [Ju] = computeJacob(Wr, AUnpruned, ti, conditioni);
    [Jp] = computeJacob(Wrpruned, APruned, ti, conditioni);
            
    if ti == tOn
        JTu = Ju;
        JTp = Jp;
    else
        JTu = JTu*Ju;
        JTp = JTp*Jp;
    end
        
end

JTu = JTu'*JTu;
JTp = JTp'*JTp;

[~, DJu] = eig(JTu);
[~, DJp] = eig(JTp);

lyapunovu = log(abs(max(diag(DJu))))/(2*(tOff - tOn));
lyapunovp = log(abs(max(diag(DJp))))/(2*(tOff - tOn));

% 
% figure;
% subplot(2,2,1);
% plot(1:length(ssefou), ssefou, 1:length(ssefop), ssefop);
% legend('Unpruned', 'Pruned');
% 
% subplot(2,2,2);
% plot(mean(pvar, 1));

mcrct = squeeze(mean(crct, 1));

merr = squeeze(mean(err, 1));
   
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trainPruned(paralleli, Lr, Li, Lo, trialType, trainCG, N, xDim)
    
Wi = 0.2*randn(Lr, Li);
Wr = 0.1*randn(Lr, Lr) + 0.5*eye(Lr, Lr);
Wo = 0.2*randn(Lo, Lr);

fixedIter = 1;

prunev = 0.7;

[Wrpruned, clampVec] = pruneNet(Wi, Wr, Wo, prunev);

[X, Y, xDim] = buildTrials(trialType, xDim);

[Wi, Wr, Wo, ssefo, iter] = trainNetwork(trainCG, Wi, Wrpruned, Wo, X, Y, trialType, fixedIter, xDim, clampVec);

weightName = sprintf('trainprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

save(weightName, 'Wi', 'Wr', 'Wo', 'ssefo', 'iter');  

trialType = 2;

[X, Y, xDim] = buildTrials(trialType, xDim);

[rWi, rWr, rWo, rssefo, riter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim, clampVec);

weightName = sprintf('retrainprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, prunev, paralleli);

save(weightName, 'rWi', 'rWr', 'rWo', 'rssefo', 'riter');  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trainUnPruned(paralleli, Lr, Li, Lo, trialType, trainCG, N, xDim)
    
Wi = 0.2*randn(Lr, Li);
Wr = 0.1*randn(Lr, Lr) + 0.5*eye(Lr, Lr);
Wo = 0.2*randn(Lo, Lr);

fixedIter = 1;

[X, Y, xDim] = buildTrials(trialType, xDim);

[Wi, Wr, Wo, ssefo, iter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim);

weightName = sprintf('trainunprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, 0, paralleli);

save(weightName, 'Wi', 'Wr', 'Wo', 'ssefo', 'iter');  

trialType = 2;

[X, Y, xDim] = buildTrials(trialType, xDim);

[rWi, rWr, rWo, rssefo, riter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim);

weightName = sprintf('retrainunprunedweightMatSCG_%d_%d_%.2f_%d.mat', N, trialType, 0, paralleli);

save(weightName, 'rWi', 'rWr', 'rWo', 'rssefo', 'riter');  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Nn, Np] = connectivityDistribution(Wr, Wrpruned)


vWrn = reshape(Wr, prod(size(Wr)), 1);
vWrp = reshape(Wrpruned, prod(size(Wr)), 1);

nDims = size(Wr, 1);

[Un, Sn, Vn] = svd(Wr);
[Up, Sp, Vp] = svd(Wrpruned);

sn = diag(Sn);
sp = diag(Sp);

zin = find(vWrn ~= 0);
zip = find(vWrp ~= 0);

Xbins = -1.5 : 0.05 : 1.5;
[Nn, X] = hist(vWrn(zin), Xbins);

[Np, X] = hist(vWrp(zip), Xbins);

% figure;
% subplot(2,2,1);
% plot(1:nDims, cumsum(diag(Sn))/sum(diag(Sn)), 1:nDims, cumsum(diag(Sp))/sum(diag(Sp)));
% legend('Unpruned', 'Pruned');
% 
% subplot(2,2,2);
% plot(X, Nn/sum(Nn), X, Np/sum(Np));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mnssef, mnpy, mnmy] = compareXVPruned(Wiretrained, Wrretrained, Woretrained, Wipruned, Wrpruned, Wopruned, trialType, xDim)


maxXV = 50;
trialCtr = 1;

for xvIter = 1 : maxXV
    
    [Xtrial, Ytrial, xDim] = buildTrials(trialType, xDim);

    nTrials = size(Xtrial, 1);

    for triali = 1 : nTrials

        [yhat] = forwardPass(Wiretrained, Wrretrained, Woretrained, squeeze(Xtrial(triali, :, :)), Ytrial(triali, :));

        ssef(trialCtr, 1) = var(yhat - Ytrial(triali, :));

        ty(trialCtr, 1, :) = yhat;

        [yhat] = forwardPass(Wipruned, Wrpruned, Wopruned, squeeze(Xtrial(triali, :, :)), Ytrial(triali, :));

        ssef(trialCtr, 2) = var(yhat - Ytrial(triali, :));

        ty(trialCtr, 2, :) = yhat;
        
        mny(trialCtr, :) = Ytrial(triali, :);
        
        trialCtr = trialCtr + 1;

    end
    
end

mnpy = squeeze(mean(ty, 1));

mnmy = squeeze(mean(mny, 1));

mnssef = mean(ssef, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mptrialc, mntrialc, dp, dn] = compareDMSPruned(Wi, Wr, Wo, Wipruned, Wrpruned, Wopruned, trialType, xDim)

[Xtrial, Ytrial, xDim] = buildTrials(trialType, xDim);

[dp, ~] = subspace(Wipruned, Wrpruned, Wopruned, Xtrial, Ytrial, xDim);
[dn, ~] = subspace(Wi, Wr, Wo, Xtrial, Ytrial, xDim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mprojn, trajContraction, lyapunov, ptrialCorrect] = probeTrajectories(Wi, Wr, Wo, Xtrial, Ytrial, xDim)

T     = size(Xtrial, 3);
nDims = size(Wr, 1);

[d, V] = subspace(Wi, Wr, Wo, Xtrial, Ytrial, xDim);

nProj = 3;

Vp = V(:, 1:nProj);

Vp(:, nProj) = Wo';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Projection of unprobed trials
pMatu       = NaN(xDim.nConditions, nProj, T);
pMatp       = NaN(xDim.nConditions, nProj, T);
yncondition = zeros(xDim.nConditions, T);
Ancondition = zeros(xDim.nConditions, nDims, T);
Apcondition = zeros(xDim.nConditions, nDims, T);

spd   = zeros(xDim.nConditions, T-1);
aMat  = zeros(xDim.nConditions, nDims);

figure;

T = size(Xtrial, 3);

for triali = 1 : xDim.nConditions
    
    [y, a] = forwardPass(Wi, Wr, Wo, squeeze(Xtrial(triali, :, :)), Ytrial(triali, :));
    
    spd(triali, :) = sum(abs(diff(a')).^2, 2);
    
    Ancondition(triali, :, :) = a;
        
    aMat(triali, :) = a(:, 37)';
    
    yncondition(triali, :) = y;
    
    pMatu(triali, :, :) = Vp'*a;
    
    subplot(2,2,triali);
    plot(1:T, squeeze(Xtrial(triali, 2:3, :)), 1:T, y);
    axis([0 Inf -1 1]);
            
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Projection of probed trials
probeStrength = 0.2;
probeBinTimes = 20;

tconditions = [1 4];

pcondition = 4;

trialProbeValue = [-1 1 -1 1];

ptrialresponse = zeros(xDim.nConditions, 1);
ptrialCorrect  = zeros(xDim.nConditions, 1);
ypcondition    = zeros(xDim.nConditions, T);

vnorminput     = zeros(xDim.nConditions, T);
vnormrecurrent = zeros(xDim.nConditions, T);

figure;
    
for triali = 1 : xDim.nConditions

    vi = rem(triali, 4);
    if vi == 0
        vi = 4;
    end
   
    binTime = probeBinTimes;

    Xi = squeeze(Xtrial(triali, :, :));

    Xi(2, binTime:(binTime+1)) = probeStrength*trialProbeValue(vi);
    
    [yhat, a, ~, ai, ar] = forwardPassNorm(Wi, Wr, Wo, Xi, Ytrial(triali, :));
    
    vnorminput(triali, :)     = ai;
    vnormrecurrent(triali, :) = ar;

    Apcondition(triali, :, :) = a;

    pMatp(triali, :, :) = Vp'*a;
    
    ypcondition(triali, :) = yhat;

    ptrialresponse(triali) = mean(yhat(52:53));
    ptrialCorrect(triali)  = 1 - abs(ptrialresponse(triali) - mean(Ytrial(triali, 52:53)));
    
    subplot(2,2,triali);
    plot(1:T, squeeze(Xi(2:3, :)), 1:T, yhat);
    axis([0 Inf -1 1]);


end

ptrialCorrect

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Force analysis
dvecr = squeeze(Ancondition(pcondition, :, :) - Apcondition(pcondition, :, :));
dvecn = squeeze(Ancondition(1, :, :)          - Apcondition(pcondition, :, :));

for ti = 2 : T
    dvect = (Wr*tanh(squeeze(Apcondition(pcondition, :, ti-1))'));
        
    rforcer(ti) = dvecr(:, ti)'*dvect/norm(dvecr(:, ti));
    rforcen(ti) = dvecn(:, ti)'*dvect/norm(dvecn(:, ti));    
    
    rforcex(ti) = (norm(dvect) - rforcer(ti))/norm(dvect);
    
end

dvc = sum((dvecr.^2), 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Time dependent Jacobian and Lyapunov

tOn  = 15;
tOff = 40;

Vj = zeros(tOff-tOn+1, nDims, nDims);
dj = zeros(tOff-tOn+1, nDims);
detJ = zeros(tOff-tOn+1, 1);

conditioni = pcondition;

tiCtr = 1;
for ti = tOn : tOff
    
    [J, Vj(tiCtr, :, :), dj(tiCtr, :)] = computeJacob(Wr, Ancondition, ti, conditioni);
    
    detJ(tiCtr) = det(J);
        
    if ti == tOn
        JT = J;
    else
        JT = JT*J;
    end
    
    tiCtr = tiCtr + 1;
    
end

JT = JT'*JT;

[~, DJ] = eig(JT);

lyapunov = diag(DJ);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Energy surface
tOn  = 10;
tOff = 52;

tOn  = probeBinTimes - 2;
tOff = probeBinTimes + 15;
nT = tOff - tOn + 1;


minx1 = min(min(pMatp(pcondition, 1, tOn:tOff))) - 1;
maxx1 = max(max(pMatp(pcondition, 1, tOn:tOff))) + 1;
minx2 = min(min(pMatp(pcondition, 2, tOn:tOff))) - 1;
maxx2 = max(max(pMatp(pcondition, 2, tOn:tOff))) + 1;

dx1 = 0.25;
dx2 = 0.25;

nx1 = floor((maxx1 - minx1)/dx1);
nx2 = floor((maxx2 - minx2)/dx2);

dSurfacep = NaN(nx1, nx2, 2);
dSurfacen = NaN(nx1, nx2, 2);

x1vs = minx1 : dx1 : maxx1-dx1;
x2vs = minx2 : dx2 : maxx2-dx2;

ptraj = squeeze(pMatp(pcondition, 1:2, tOn:tOff));
ntraj = squeeze(pMatu(pcondition, 1:2, tOn:tOff));

for x1i = 1 : nx1
    for x2i = 1 : nx2
        
        x1 = minx1 + (x1i - 1)*dx1;
        x2 = minx2 + (x2i - 1)*dx2;
        
        xmat = repmat([x1; x2], 1, nT);
        
        dxpv = xmat - ptraj;
        dxnv = xmat - ntraj;
        
        dxp = sum(abs(dxpv));
        dxn = sum(abs(dxnv));
        
        [d1vp, mnip] = min(dxp);
        [~, mnin] = min(dxn);
        
        if d1vp > 6*dx1 
            continue;
        end
        
        fpp = (Apcondition(pcondition, :, mnip + tOn-1))';
        fpn = (Ancondition(pcondition, :, mnin + tOn-1))';
        
        fppt = fpp + Vp(:, 1:2)*dxpv(:, mnip);
        fpnt = fpn + Vp(:, 1:2)*dxnv(:, mnin);

        dap = Wr*tanh(fppt) - fppt;
        dan = Wr*tanh(fpnt) - fpnt;
        
        dapl = Vp'*dap;
        danl = Vp'*dan;

        dSurfacep(x1i, x2i, :) = -dapl(1:2);
        dSurfacen(x1i, x2i, :) = -danl(1:2);
        
    end
end

% figure
% subplot(2,2,1);
% 
% surf(x1vs, x2vs, squeeze(dSurfacep(:, :, 1))');
% view(0, 90);
% 
% subplot(2,2,2);
% surf(x1vs, x2vs, squeeze(dSurfacep(:, :, 2))');
% view(0, 90);

% [dSurfaceps1] = smooth2dcg(squeeze(dSurfacep(:, :, 1)), 0.25, 5, 5);
% [dSurfaceps2] = smooth2dcg(squeeze(dSurfacep(:, :, 2)), 0.25, 5, 5);
% [dSurfacens1] = smooth2dcg(squeeze(dSurfacen(:, :, 1)), 0.25, 5, 5);
% [dSurfacens2] = smooth2dcg(squeeze(dSurfacen(:, :, 2)), 0.25, 5, 5);

[dSurfaceps1] = squeeze(dSurfacep(:, :, 1));
[dSurfaceps2] = squeeze(dSurfacep(:, :, 2));
[dSurfacens1] = squeeze(dSurfacen(:, :, 1));
[dSurfacens2] = squeeze(dSurfacen(:, :, 2));

denergyp1 = zeros(nx1, nx2);
denergyp2 = zeros(nx1, nx2);
denergyn1 = zeros(nx1, nx2);
denergyn2 = zeros(nx1, nx2);

for x1i = 2 :  nx1
    for x2i = 2 : nx2
        
        if isnan(dSurfaceps1(x1i-1, x2i))
            denergyp1(x1i, x2i) = denergyp1(x1i-1, x2i);
        else
            denergyp1(x1i, x2i) = denergyp1(x1i-1, x2i) + dSurfaceps1(x1i-1, x2i);
        end
        
        if isnan(dSurfaceps2(x1i, x2i-1))
            denergyp2(x1i, x2i) = denergyp2(x1i, x2i-1);
        else
            denergyp2(x1i, x2i) = denergyp2(x1i, x2i-1) + dSurfaceps2(x1i, x2i-1);
        end
        
        if isnan(dSurfacens1(x1i-1, x2i))
            denergyn1(x1i, x2i) = denergyn1(x1i-1, x2i);
        else
            denergyn1(x1i, x2i) = denergyn1(x1i-1, x2i) + dSurfacens1(x1i-1, x2i);
        end
        
        if isnan(dSurfacens2(x1i, x2i-1))
            denergyn2(x1i, x2i) = denergyn2(x1i, x2i-1);
        else
            denergyn2(x1i, x2i) = denergyn2(x1i, x2i-1) + dSurfacens2(x1i, x2i-1);            
        end
        
    end
end

% subplot(2,2,3);
% surf(x1vs, x2vs, denergyp1');
% view(0, 90);
% 
% subplot(2,2,4);
% surf(x1vs, x2vs, denergyp2');
% view(0, 90);


% figure;
% surf(x1vs, x2vs, (denergyp1 + denergyp2)', 'FaceAlpha', 0.6);
% shading interp;
% 
% hold on;
% 
% view(0, 90);
% plot(squeeze(pMatp(pcondition, 1, tOn:tOff))', squeeze(pMatp(pcondition, 2, tOn:tOff))', 'LineWidth', 3);
% text(squeeze(pMatp(pcondition, 1, tOn)),  squeeze(pMatp(pcondition, 2, tOn)),'s', 'Fontsize', 20);
% text(squeeze(pMatp(pcondition, 1, probeBinTimes)),  squeeze(pMatp(pcondition, 2, probeBinTimes)),'P', 'Fontsize', 20);
%     
%       
% figure;
% surf(x1vs, x2vs, (denergyn1 + denergyn2)', 'FaceAlpha', 0.6);
% shading interp;
% view(0, 90);
% hold on;
% plot(squeeze(pMatu(pcondition, 1, tOn:tOff))', squeeze(pMatu(pcondition, 2, tOn:tOff))', 'LineWidth', 3);
% text(squeeze(pMatp(pcondition, 1, tOn)),  squeeze(pMatp(pcondition, 2, tOn)),'s', 'Fontsize', 20);
% text(squeeze(pMatp(pcondition, 1, probeBinTimes)),  squeeze(pMatp(pcondition, 2, probeBinTimes)),'P', 'Fontsize', 20);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Flow field
tOn  = probeBinTimes - 2;
% tOff = probeBinTimes + 15;
tOff = probeBinTimes + 10;

fpCondition = 4;
x3 = 0;


Xi = squeeze(Xtrial(fpCondition, :, :));


ds = 0.05;

pctr = 0;
for ti = tOn : tOff
    
    tpOn = pctr + 1;

    for x1 = -ds : ds : ds
        for x2 = -ds : ds : ds

            pctr = pctr + 1;

            fpp = (Apcondition(fpCondition, :, ti))';
            fpn = (Ancondition(fpCondition, :, ti))';

            %%% 2-d perturb around trajectory on PCA axes
            dx = Vp(:, 1:3)*[x1; x2; x3];

            fppt = fpp + dx;
            fpnt = fpn + dx;

            %%% compute derivative in full latent space
            dap = Wi*Xi(:, ti) + Wr*tanh(fppt) - fppt;
            dan = Wi*Xi(:, ti) + Wr*tanh(fpnt) - fpnt;

            spfp(pctr) = sum(abs(dap).^2);

            nxi(pctr) = norm([x1; x2; x3]);

            %%% projection of derivative into PCA space
            dapt(pctr, :) = Vp'*dap;
            dxpt(pctr, :) = squeeze(pMatp(fpCondition, :, ti)) + [x1 x2 x3];
            
            dant(pctr, :) = Vp'*dan;
            dxnt(pctr, :) = squeeze(pMatu(fpCondition, :, ti)) + [x1 x2 x3];
            
%             if x1 == 0 && x2 == 0
%                 fpnp1 = squeeze(pMatu(fpCondition, :, ti+1));
% 
%                 [fpnp1' dant(pctr, :)']
%                 
%                 [dan+fpnt squeeze(Ancondition(fpCondition, :, ti+1))']
% 
%                 fprintf('');
%                 
%             end
        end
    end
    
    %%% find point on mean trajectory
    [~, pi] = min(nxi(tpOn:pctr));
    
    pcpi = tpOn + pi - 1;
    
    %%% derivative on mean trajectory
    trajDap = dapt(pcpi, :);
    trajDan = dant(pcpi, :);
    
    %%% normalized
    trajDap = trajDap/norm(trajDap);
    trajDan = trajDan/norm(trajDan);
    
    piCtr = 0;
    for tpi = tpOn : pctr
        
        if tpi == pcpi
            continue;
        end
        
        piCtr = piCtr + 1;
        
        %%% Is enpoint closer or further than beginning after perturb
        eptpD(piCtr) = nxi(tpi) - norm((dxpt(tpi, :)' + dapt(tpi, :)') - (dxpt(pcpi, :)' + dapt(pcpi, :)'));
        eptnD(piCtr) = nxi(tpi) - norm((dxnt(tpi, :)' + dant(tpi, :)') - (dxnt(pcpi, :)' + dant(pcpi, :)'));
        
        projp(piCtr) = norm(dapt(tpi, :)' - (trajDan*dapt(tpi, :)')*trajDan')/norm(dapt(tpi, :));
        projn(piCtr) = norm(dant(tpi, :)' - (trajDan*dant(tpi, :)')*trajDan')/norm(dant(tpi, :));
                
%         dxvec = dant(tpi, :)' - (trajDan*dant(tpi, :)')*trajDan';
%         pvec1 = [dxnt(tpi, 1:2); dxnt(tpi, 1:2) + dant(tpi, 1:2)];
%         pvec2 = [dxnt(pcpi, 1:2); dxnt(pcpi, 1:2) + dant(pcpi, 1:2)];        
%         pvec3 = [0 0; dxvec(1:2)'];
%         pvec4 = [dant(tpi, 1:2); dant(tpi, 1:2)-dxvec(1:2)'];
%         
%         line(pvec1(:, 1), pvec1(:,2), 'color', 'red');
%         line(pvec2(:, 1), pvec2(:,2), 'color', 'blue');
%         line(pvec3(:, 1), pvec3(:, 2), 'color', 'green');
%         line(pvec4(:,1), pvec4(:,2));
        
%         fprintf('');
                
    end
    
%     mprojp(ti) = mean(projp);
%     mprojn(ti) = mean(projn);    

    mprojp(ti) = mean(eptpD);
    mprojn(ti) = mean(eptnD);    

    dcum = d/sum(d);
    
    wdim = repmat(dcum(1:nProj)', pctr-tpOn+1, 1);

    ep = (dapt(tpOn:pctr, :) + dxpt(tpOn:pctr, :));
    en = (dant(tpOn:pctr, :) + dxnt(tpOn:pctr, :));
    
    bp = dxnt(tpOn:pctr, :);

    %%% Squared difference between probed and unprobed trajectory at end
    spvar(ti) = sqrt(sum(sum((abs(ep - repmat(ep(pi, :), pctr-tpOn+1, 1)).^2).*wdim)));  
    snvar(ti) = sqrt(sum(sum((abs(en - repmat(en(pi, :), pctr-tpOn+1, 1)).^2).*wdim)));  
    
    bpvar(ti) = sqrt(sum(sum((abs(bp - repmat(bp(pi, :), pctr-tpOn+1, 1)).^2).*wdim)));  
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2-d
figure;

hold on;

for conditionvi = 2:length(tconditions)
    
    conditioni = tconditions(conditionvi);
    
    plot(squeeze(pMatu(conditioni, 1, tOn:tOff))', squeeze(pMatu(conditioni, 2, tOn:tOff))', 'LineWidth', 3);
      
end

%%% When we probe condition 4 it looks like condition 1
plot(squeeze(pMatp(pcondition, 1, tOn:tOff))', squeeze(pMatp(pcondition, 2, tOn:tOff))', 'LineWidth', 3);
 
xlabel('PC1');
ylabel('PC2');

for pti = 1 : pctr
   
    %%% Around probed trajectory
%     lx = [dxpt(pti, 1) (dxpt(pti, 1) + dapt(pti, 1))];
%     ly = [dxpt(pti, 2) (dxpt(pti, 2) + dapt(pti, 2))];
%     
%     line(lx, ly);
%     text(dxpt(pti, 1), dxpt(pti, 2), 'o', ...
%         'HorizontalAlignment', 'center');
    
    %%% Around unprobed trajectory
    lx = [dxnt(pti, 1) (dxnt(pti, 1) + dant(pti, 1))];
    ly = [dxnt(pti, 2) (dxnt(pti, 2) + dant(pti, 2))];
    
    line(lx, ly, 'Color',0.5*[0 0.447 0.741]);
    text(dxnt(pti, 1), dxnt(pti, 2), 'o', ...
        'HorizontalAlignment', 'center');
        
end

text(squeeze(pMatp(pcondition, 1, tOn)),  squeeze(pMatp(pcondition, 2, tOn)),'S', 'Fontsize', 20);
text(squeeze(pMatp(pcondition, 1, probeBinTimes)),  squeeze(pMatp(pcondition, 2, probeBinTimes)),'P', 'Fontsize', 20);
text(squeeze(pMatp(pcondition, 1, tOff)),  squeeze(pMatp(pcondition, 2, tOff)),'E', 'Fontsize', 20);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 3-d
figure;
hold on;
for conditionvi = 2:length(tconditions)
    
    conditioni = tconditions(conditionvi);
    
    plot3(squeeze(pMatu(conditioni, 1, tOn:tOff))', squeeze(pMatu(conditioni, 2, tOn:tOff))', ...
          squeeze(pMatu(conditioni, 3, tOn:tOff))', 'LineWidth', 3);
      
end

%%% When we probe condition 4 it looks like condition 1
plot3(squeeze(pMatp(pcondition, 1, tOn:tOff))', squeeze(pMatp(pcondition, 2, tOn:tOff))', ...
      squeeze(pMatp(pcondition, 3, tOn:tOff))', 'LineWidth', 3);
 
xlabel('PC1');
ylabel('PC2');
zlabel('dR');

for pti = 1 : pctr
   
    lx = [dxpt(pti, 1) (dxpt(pti, 1) + dapt(pti, 1))];
    ly = [dxpt(pti, 2) (dxpt(pti, 2) + dapt(pti, 2))];
    lz = [dxpt(pti, 3) (dxpt(pti, 3) + dapt(pti, 3))];
    
    line(lx, ly, lz, 'Color',0.5*[0.850 0.325 0.098]);
    text(dxpt(pti, 1), dxpt(pti, 2), dxpt(pti, 3), 'o', ...
        'HorizontalAlignment', 'center');
    
    lx = [dxnt(pti, 1) (dxnt(pti, 1) + dant(pti, 1))];
    ly = [dxnt(pti, 2) (dxnt(pti, 2) + dant(pti, 2))];
    lz = [dxnt(pti, 3) (dxnt(pti, 3) + dant(pti, 3))];
    
    line(lx, ly, lz, 'Color',0.5*[0 0.447 0.741]);
    text(dxnt(pti, 1), dxnt(pti, 2), dxnt(pti, 3), 'o', ...
        'HorizontalAlignment', 'center');
        
end

text(squeeze(pMatu(tconditions(2), 1, 40)), squeeze(pMatu(tconditions(2), 2, 40)), squeeze(pMatu(tconditions(2), 3, 40)),'c21', 'HorizontalAlignment', 'center', 'Fontsize', 20);
text(squeeze(pMatp(pcondition, 1, 40)), squeeze(pMatp(pcondition, 2, 40)), squeeze(pMatp(pcondition, 3, 40)),'c2p', 'HorizontalAlignment', 'center', 'Fontsize', 20);
text(squeeze(pMatp(pcondition, 1, probeBinTimes)), squeeze(pMatp(pcondition, 2, probeBinTimes)), squeeze(pMatp(pcondition, 3, probeBinTimes)),'P', 'HorizontalAlignment', 'center', 'Fontsize', 20);

  

tOn  = 12;
tOff = 52;

figure;
subplot(2,2,1);
plot(tOn:tOff, squeeze(pMatu(tconditions(1), 1, tOn:tOff))', ...
     tOn:tOff, squeeze(pMatu(tconditions(2), 1, tOn:tOff))', ...
     tOn:tOff, squeeze(pMatp(pcondition, 1, tOn:tOff))', 'LineWidth', 2);
title('PC1');
 
text(13, min(squeeze(pMatu(tconditions(1), 1, tOn:tOff))), 'C1', 'HorizontalAlignment', 'center');
text(probeBinTimes, min(squeeze(pMatu(tconditions(1), 1, tOn:tOff))), 'P', 'HorizontalAlignment', 'center');
text(40, min(squeeze(pMatu(tconditions(1), 1, tOn:tOff))), 'C2', 'HorizontalAlignment', 'center');
text(52, min(squeeze(pMatu(tconditions(1), 1, tOn:tOff))), 'R', 'HorizontalAlignment', 'center');
 
subplot(2,2,2);
plot(tOn:tOff, squeeze(pMatu(tconditions(1), 2, tOn:tOff))', ...
     tOn:tOff, squeeze(pMatu(tconditions(2), 2, tOn:tOff))', ...
     tOn:tOff, squeeze(pMatp(pcondition, 2, tOn:tOff))', 'LineWidth', 2);
title('PC2');
 
subplot(2,2,3);
plot(tOn:tOff, squeeze(pMatu(tconditions(1), 3, tOn:tOff))', ...
     tOn:tOff, squeeze(pMatu(tconditions(2), 3, tOn:tOff))', ...
     tOn:tOff, squeeze(pMatp(pcondition, 3, tOn:tOff))', 'LineWidth', 2);
legend('Cond1', 'Cond4', 'Probe');
title('Wo');

subplot(2,2,4);
plot(1:T, yncondition(tconditions, :), 1:T, ypcondition(4, :), 'LineWidth', 2);
legend('C1', 'C4', 'P4');

trajContraction = snvar./bpvar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [msurf] = smooth2dcg(z, b, smx, smy)

minx = 1;
maxx = size(z, 1);
miny = 1;
maxy = size(z, 2);

msurf = NaN(size(z, 1), size(z, 2));

for xi = minx : maxx
    
    for yi = miny : maxy
        
        dtxy = [xi; yi];
        
        xsmin = xi - smx;
        xsmax = xi + smx;
        if xsmin < 1    xsmin = 1;    end
        if xsmax > maxx xsmax = maxx; end;
        
        ysmin = yi - smy;
        ysmax = yi + smy;
        if ysmin < 1    ysmin = 1;    end;
        if ysmax > maxy ysmax = maxy; end;      
                
        vsum = 0;
        psum = 0;
        for xs = xsmin : xsmax
            for ys = ysmin : ysmax
                 
                if isnan(z(xs, ys))
                    continue;
                end
                
                smpxy = [xs; ys];
                
                dst = norm(dtxy - smpxy);
                            
                prb = exp(-(dst^2)/b^2);
                
                vsum = vsum + prb*z(xs, ys);
                psum = psum + prb;
                
            end
                        
        end
        
        msurf(xi, yi) = vsum/psum;    
        
    end
        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d, V] = subspace(Wi, Wr, Wo, Xtrial, Ytrial, xDim)

T = size(Xtrial, 3);
nDims = size(Wr, 1);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%PCA axes
Tuse = T-10;

Aft = NaN(nDims, xDim.nConditions*Tuse);

AftCondition = NaN(xDim.nConditions, nDims, T);

for triali = 1 : xDim.nConditions
    
    [~, a] = forwardPass(Wi, Wr, Wo, squeeze(Xtrial(triali, :, :)), Ytrial(triali, :));
    
    sbin = (triali-1)*Tuse + 1;
    ebin = sbin + Tuse - 1;
    Aft(:, sbin:ebin) = a(:, 11:end);
    
    AftCondition(triali, :, :) = a;
            
end

cvA = cov(Aft');

[V, D] = eig(cvA);

d = diag(D);

d = d(end:-1:1);
V = V(:, end:-1:1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mptrialCorrect = probePerformance(Wi, Wr, Wo, Xtrial, Ytrial, xDim)


% probeStrength = 0 : 0.2 : 1.2;
probeStrength = 0.8;

probeBinTimes = 15 : 4 : 38;

trialProbeValue = [-1 1 -1 1];
    
for triali = 1 : size(Xtrial, 1)

    vi = rem(triali, 4);
    if vi == 0
        vi = 4;
    end
    
    for probeBini = 1 : length(probeBinTimes)
        
        binTime = probeBinTimes(probeBini);
        
        Xi = squeeze(Xtrial(triali, :, :));
        
        for probei = 1 : length(probeStrength)

            Xi(2, binTime:(binTime+1)) = probeStrength(probei)*trialProbeValue(vi);

            [yhat] = forwardPass(Wi, Wr, Wo, Xi, Ytrial(triali, :));

            ptrialresponse(probeBini, probei, triali) = mean(yhat(42:43));
            ptrialCorrect(probeBini, probei, triali)  = 1 - abs(ptrialresponse(probeBini, probei, triali) - mean(Ytrial(triali, 42:43)));

        end
        
    end
    
end

mptrialCorrect = squeeze(mean(double(ptrialCorrect > 0), 3))';

squeeze(mean(double(ptrialCorrect > 0), 3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wr, clampVec] = pruneNet(Wi, Wr, Wo, prunev)

nPrune = floor(prunev*numel(Wr));

fprintf('Pruning %d of %d\n', nPrune, numel(Wr));

pruneType = 0;

if pruneType == 1
    
    nrowPrune = floor(nPrune/size(Wr, 1));
    
    for rowi = 1 : size(Wr, 1)
    
        pruneindex = randperm(size(Wr, 2));

        Wr(rowi, pruneindex(1:nrowPrune)) = 0;
    
    end
    
end

vWr = reshape(Wr, numel(Wr), 1);

HvweightSalience = abs(vWr);

[~, w_index] = sort(HvweightSalience, 'ascend');

Wr(w_index(1:nPrune)) = 0;

WrclampVec = ones(numel(Wr), 1);
WrclampVec(w_index(1:nPrune)) = 0;

totalElements = numel(Wi) + numel(Wr) + numel(Wo);
clampVec = ones(totalElements, 1);
clampVec(numel(Wi)+1:numel(Wi) + numel(Wr)) = WrclampVec;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J, Vj, dj] = computeJacob(Wr, Afcondition, fp, conditioni)

nDims = size(Wr, 1);

J = NaN(nDims, nDims);

for w1 = 1 : nDims
    for w2 = 1 : nDims

        J(w1, w2) = Wr(w1, w2)*dftanh(squeeze(Afcondition(conditioni, w2, fp(1))));

    end

end

[Vj, Dj] = eig(J);

dj = diag(Dj);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wi, Wr, Wo, ssefo, iter] = trainNetwork(trainCG, Wi, Wr, Wo, X, Y, trialType, fixedIter, xDim, varargin)
   
N = size(Wr, 1);

if length(varargin) == 1
    clampVec = varargin{1};
else
    totalElements = numel(Wi) + numel(Wr) + numel(Wo);
    clampVec = ones(totalElements, 1);    
end
   
T = size(Y, 2);

if trainCG == 1
    
    if fixedIter == -1

        maxBPIter = 100;
        ssefo = NaN(maxBPIter, 1);
        
    else
        
%         maxBPIter = 20;
        maxBPIter = 50; %%% Only for trainPrune analysis
        
    end

    for iter = 1 : maxBPIter

        [dWi, dWr, dWo, sseft] = backProp(Wi, Wr, Wo, X, Y);

        ssefo(iter) = sseft;

        dEdtheta = vecWeights(dWi, dWr, dWo);

        [Wi, Wr, Wo, converged] = conjGrad(dEdtheta, Wi, Wr, Wo, X, Y, sseft, fixedIter, clampVec, xDim); 

        if converged == 1 && fixedIter == -1
            break;
        end
        
    end
    
else
    
    [Wi, Wr, Wo] = trainAdam(Wi, Wr, Wo, X, Y, clampVec);
    
end

% [X, Y, xDim] = buildTrials(trialType, xDim);
% 
% figure;
% 
% if trialType == 1
% 
%     for triali = 1 : 4
% 
%         [yhat] = forwardPass(Wi, Wr, Wo, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));
% 
%         subplot(2,2,triali);
%         plot(1:T, squeeze(X(triali,2, :)), 1:T, squeeze(X(triali, 3, :)), ...
%             1:T, yhat, 1:T, squeeze(mean(Y(triali:4:end, :), 1)));
% 
%     end
%     
% else
%     
%     [yhat] = forwardPass(Wi, Wr, Wo, squeeze(X(1, :, :)), squeeze(Y(1, :)));
% 
%     plot(1:T, yhat, 1:T, squeeze(Y(1, :)));
%         
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wi, Wr, Wo] = trainAdam(Wi, Wr, Wo, X, Y, clampVec)

maxBPIter = 5000;
beta1     = 0.9;
beta2     = 0.999;
epsilon   = 10^(-8);

lre0   = 0.07;
lrtau  = 1000;
lretau = 0.001;

%%% Weight decay parameter
weightDecay = 1e-4;

ssefo = NaN(maxBPIter, 1);

normDistCtr = 1;
normRun = 30;
normDist = NaN(normRun, 1);

ndiststd = 1;
ndistmn  = 1;

rWeights = numel(Wi) + 1 : numel(Wi) + numel(Wr);

m = 0;
v = 0;

sparseM = clampVec(1) == -1;

for iter = 1 : maxBPIter
    
    [dWi, dWr, dWo, ssef] = backProp(Wi, Wr, Wo, X, Y);

    ssefo(iter) = ssef;
       
    dEdtheta = vecWeights(dWi, dWr, dWo);
    
    if sparseM == 1
        vWk = vecWeights(Wi, Wr, Wo);
        weightSalience = abs(dEdtheta(rWeights).*vWk(rWeights));
        [~, weightIndex] = sort(weightSalience, 'ascend');
        nPrune = floor(0.05*numel(dEdtheta));
        clampVec = ones(numel(dEdtheta), 1);
        clampVec(rWeights(weightIndex(1:nPrune))) = 0;
    end
        
    m = beta1*m + (1 - beta1)*dEdtheta;
    v = beta2*v + (1 - beta2)*(dEdtheta.^2);
    
    mdb = m/(1 - beta1.^(iter));
    vdb = v/(1 - beta2.^(iter));
    
    dEm = mdb./(sqrt(vdb) + epsilon);   
           
    %%% Norm clipping
    dEnorm = norm(dEm);
 
    normDist(normDistCtr) = dEnorm;
    normDistCtr = normDistCtr + 1;
    
    if normDistCtr > normRun
       normDistCtr = 1;
    end
    
    if iter > 100

        ndiststd = nanstd(normDist);
        ndistmn  = nanmean(normDist);

        if dEnorm > 3*ndiststd + ndistmn
            dEm = 3*ndiststd*dEm/dEnorm;
            if normDistCtr > 1
                normDistCtr = normDistCtr - 1;
            end
        end


    end
    
    %%% Prune
    dEm = dEm.*clampVec;

    [dWiq, dWrq, dWoq] = matWeights(dEm);
    
    %%% learning rate decay
    if iter < lrtau
        lambda = iter/lrtau;
        lr = (1 - lambda)*lre0 + lambda*lretau;
    else
        lr = lretau;
    end
    
    Wi = Wi - lr*dWiq - weightDecay*Wi;
    Wr = Wr - lr*dWrq - weightDecay*Wr;
    Wo = Wo - lr*dWoq - weightDecay*Wo;
        
    if rem(iter,20) == 0
        fprintf('%d/%d %.5f %.4f %.3f %.3f\n', ...
            iter, maxBPIter, ssefo(iter), dEnorm, lr, ndistmn+ 2*ndiststd);        
    end
    
    if ssefo(iter) < .01
        break;
    end
    
                
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wi, Wr, Wo, converged] = conjGrad(dEdtheta, Wi, Wr, Wo, X, Y, ssef0, fixedIter, clampVec, xDim)

vWk = vecWeights(Wi, Wr, Wo);
    
matSize = [size(Wi, 2) size(Wr, 1) size(Wo, 1)];

if fixedIter == -1 
    maxCGIter = numel(vWk);
else
%     maxCGIter = min([fixedIter numel(vWk)]);
    maxCGIter = min([numel(vWk)]);
end

sigma = 5e-5;
lr = 0.001;
weightDecay = 0;

lambdak = 5e-7;

pk = -dEdtheta.*clampVec;
rk = -dEdtheta.*clampVec;
Ek = ssef0;
success   = 1;
failCtr   = 0;
converged = 0;
dfEk      = 0;
sparseM   = clampVec(1) == -1;

rWeights = numel(Wi) + 1 : numel(Wi) + numel(Wr);

EkN       = min([300 ceil(maxCGIter*0.8)]);
EkCtr     = 1;
dfEkv     = .01*ones(EkN, 1);

for cgIter = 1 : maxCGIter

    npk = norm(pk);

    if success == 1
        
        sigmak = sigma/npk;

        vWkd = vWk + sigmak*pk;

        [Wik, Wrk, Wok] = matWeights(vWkd, matSize);

        [dWik, dWrk, dWok] = backProp(Wik, Wrk, Wok, X, Y);

        [dEdthetak] = vecWeights(dWik, dWrk, dWok);

        sk = (dEdthetak - dEdtheta)/sigmak;

        deltak = pk'*sk;
        
    end
    
    deltak = deltak + lambdak*(npk^2);
    
    if deltak < 0
                
        lambdak = 2*(lambdak - deltak/(npk^2));
        deltak = deltak + lambdak*(npk^2);
        
    end
    
    muk = pk'*rk;
    alphak = muk/deltak;    
            
    tvWk = vWk + alphak*pk;
        
    [tWi, tWr, tWo] = matWeights(tvWk, matSize);
        
    [dWi, dWr, dWo, ssef] = backProp(tWi, tWr, tWo, X, Y);
    
    gDeltak = 2*deltak*(Ek - ssef)/(muk^2);
    
    if gDeltak >= 0
        
        vWk = tvWk - weightDecay*vWk;
                
        rkp1 = vecWeights(dWi, dWr, dWo);
        
        dEdtheta = rkp1;
        
        rkp1 = -rkp1;
                        
        success = 1;
        failCtr = 1;

        nrkp1 = norm(rkp1);

        betak = (nrkp1^2 - rkp1'*rk)/muk;

        pk = rkp1 + betak*pk;
        
        if sparseM == 1
            weightSalience = abs(rkp1(rWeights).*vWk(rWeights));
            [~, weightIndex] = sort(weightSalience, 'ascend');
            nPrune = floor(0.05*numel(vWk));
            clampVec = ones(numel(vWk), 1);
            clampVec(rWeights(weightIndex(1:nPrune))) = 0;
        end
        
        %%% for retraining after pruning
        pk = pk.*clampVec;

        if gDeltak > 0.75
            
            lambdak = 0.25*lambdak;
            
        end
                                
        rk = rkp1;
        
        Ekm1 = Ek;
        
        Ek = ssef;
        
    else
        
        success = 0;
        
        failCtr = failCtr + 1;
        
        if failCtr > 10
            fprintf('Failed, taking gradient step\n');
            fprintf('CG %d sse %.6f delta %.3e Delta %.3e lambda %.3e alpha %.4f success %d\n', ...
                cgIter, Ek, deltak, gDeltak, lambdak, alphak, success);

            pk = rk;

            vWk = vWk + lr*pk;
            failCtr = 1;
            
        end
        
    end
    
    if gDeltak < 0.25
            
        lambdak = lambdak + (deltak*(1 - gDeltak)/npk^2);
    
    end 
    
    if success == 1
        dfEk = Ekm1 - Ek;
        
        if EkCtr <= EkN
            dfEkv(EkCtr) = dfEk;
            EkCtr = EkCtr + 1;
        else
            EkCtr = 1;
            dfEkv(EkCtr) = dfEk;
        end
        
        mndfEk = mean(dfEkv);
        
    end
    
    if success == 1 && mndfEk < 0.00000001 && Ekm1 < (xDim.sigmay^2)*1.05

        fprintf('Cgiter %d Ek %.5f Ek-1 %.5f %.6f\n', cgIter, Ek, Ekm1, mndfEk);

        [Wi, Wr, Wo] = matWeights(vWk);
        converged = 1;
        return;
        
    end
    
    if rem(cgIter, 20) == 0
        fprintf('CG %d sse %.6f mnDF %.6f delta %.3e Delta %.3f lambda %.3e alpha %.4f success %d\n', ...
            cgIter, Ek, mndfEk, deltak, gDeltak, lambdak, alphak, success);
    end
    
end

[Wi, Wr, Wo] = matWeights(vWk);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dWi, dWr, dWo, ssef] = backProp(Wi, Wr, Wo, X, Y)

tdWi = zeros(size(X, 1), size(Wi, 1), size(Wi, 2));
tdWr = zeros(size(X, 1), size(Wr, 1), size(Wr, 2));
tdWo = zeros(size(X, 1), size(Wo, 1), size(Wo, 2));

tssefo = zeros(size(X, 1), 1);

for triali = 1 : size(X, 1) 
        
    [yhat, a, z] = forwardPass(Wi, Wr, Wo, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));

    [dWi, dWr, dWo] = backWardPass(Wr, Wo, yhat, a, z, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));

    if sum((size(dWo) - size(Wo)).^2) ~= 0
        dWo = dWo';
    end

    tdWi(triali, :, :) = dWi;
    tdWr(triali, :, :) = dWr;
    tdWo(triali, :, :) = dWo;

    tssefo(triali) = sum((yhat - Y(triali, :)).^2);

end

nPoints = size(X, 1)*size(X, 3);

dWi = squeeze(sum(tdWi, 1))/nPoints;
dWr = squeeze(sum(tdWr, 1))/nPoints;
dWo = squeeze(sum(tdWo, 1))/nPoints;

ssef = sum(tssefo)/nPoints;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [yhat, a, z] = forwardPass(Wi, Wr, Wo, Xi, y)

T = size(y, 2);
Lr = size(Wr, 1);
Lo = size(Wo, 1);

yhat = NaN(Lo, T);

a    = zeros(Lr, T);
z    = zeros(Lr, T);
z0   = zeros(Lr, 1);
z(:, 1) = z0;

for t = 1 : length(y)

    %%% Forward pass
    %%% Linear term
    if t > 1
        a(:, t) = Wi*Xi(:, t) + Wr*z(:, t-1);
    else
        a(:, t) = Wi*Xi(:, t);
    end

    %%% Nonlinear transfer
    z(:,t) = tanh(a(:,t));

    %%% Output
    yhat(:, t) = Wo*z(:,t);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [yhat, a, z, ai, ar] = forwardPassNorm(Wi, Wr, Wo, Xi, y)

T = size(y, 2);
Lr = size(Wr, 1);
Lo = size(Wo, 1);

yhat = NaN(Lo, T);

a    = zeros(Lr, T);
z    = zeros(Lr, T);
z0   = zeros(Lr, 1);
z(:, 1) = z0;

ai   = zeros(1, T);
ar   = zeros(1, T);

for t = 1 : length(y)

    %%% Forward pass
    %%% Linear term
    if t > 1
        
        ai(t) = norm(Wi*Xi(:, t));
        ar(t) = norm(Wr*z(:, t-1));
        
        a(:, t) = Wi*Xi(:, t) + Wr*z(:, t-1);
        
    else
        a(:, t) = Wi*Xi(:, t);
    end

    %%% Nonlinear transfer
    z(:,t) = tanh(a(:,t));

    %%% Output
    yhat(:, t) = Wo*z(:,t);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dWi, dWr, dWo] = backWardPass(Wr, Wo, yhat, a, z, Xi, y)

T = size(y, 2);
Lr = size(Wr, 1);
Lo = size(Wo, 1);
Li = size(Xi, 1);

drt   = zeros(Lr, T+1);

dedwo = zeros(T, Lo, Lr);
dedwi = zeros(T, Lr, Li);
dedwr = zeros(T, Lr, Lr); 

dy = 2*(yhat - y);

for t = length(y) : -1 : 1

    %%% Backward pass        
    dft = dftanh(a(:,t));
    
    drt(:, t) = dft.*((Wo'*dy(:,t)) + (Wr'*drt(:,t+1)));

    dedwo(t, :, :) = dy(:,t)*z(:,t)';
    dedwi(t, :, :) = drt(:,t)*Xi(:,t)';

    if t > 1
        dedwr(t, :, :) = drt(:,t)*z(:,t-1)';
    end

end

dWi = (squeeze(sum(dedwi, 1)));
dWr = (squeeze(sum(dedwr, 1)));
dWo = (squeeze(sum(dedwo, 1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta = vecWeights(mWi, mWr, mWo)

vWi = reshape(mWi, numel(mWi), 1);
vWr = reshape(mWr, numel(mWr), 1);
vWo = reshape(mWo, numel(mWo), 1);

theta = [vWi; vWr; vWo];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mWi, mWr, mWo] = matWeights(theta, varargin)

global Li;
global Lr;
global Lo;

if size(varargin) > 0
   
    Li = varargin{1}(1);
    Lr = varargin{1}(2);
    Lo = varargin{1}(3);
    
end

indis = 1;
indie = Li*Lr;

indrs = indie + 1;
indre = indrs + Lr*Lr - 1;

indos = indre + 1;
indoe = length(theta);

vWi = theta(indis:indie);
vWr = theta(indrs:indre);
vWo = theta(indos:indoe);

mWi = reshape(vWi, Lr, Li);
mWr = reshape(vWr, Lr, Lr);
mWo = reshape(vWo, Lo, Lr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dh = dftanh(x)

dh = 1 - tanh(x).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weightSalience = HvSalience(Wi, Wr, Wo, trialType, xDim)

epsilon = 10e-7;

Wsq = Wr.^2;

[X, Y, xDim] = buildTrials(trialType, xDim);

%%%%%%%%%%%%% Positive
Wrvp = Wr + epsilon*Wsq;

[~, dWrp, ~] = backProp(Wi, Wrvp, Wo, X, Y);


%%%%%%%%%%%%% Negative
Wrvn = Wr - epsilon*Wsq;

[~, dWrn, ~] = backProp(Wi, Wrvn, Wo, X, Y);


vdWrp = reshape(dWrp, numel(dWrp), 1);
vdWrn = reshape(dWrn, numel(dWrn), 1);

weightSalience = (vdWrp - vdWrn)/(2*epsilon);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weightSalience = numHessian(Wi, Wr, Wo, trialType, xDim)

epsilon = 10e-7;

[X, Y, xDim] = buildTrials(trialType, xDim);

N = numel(Wr);

H = zeros(N, 1);

parfor w1 = 1 : N
    
    fprintf('%d of %d\n', w1, N);
    
    w2 = w1;
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    Wu = Wr;
    Wu(w1) = Wu(w1) + epsilon;
    Wu(w2) = Wu(w2) + epsilon;

    tssefo = zeros(size(X, 1), 1);

    for triali = 1 : size(X, 1) 

        yhat = forwardPass(Wi, Wu, Wo, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));
        tssefo(triali) = sum((yhat - Y(triali, :)).^2);

    end

    Ew1 = sum(tssefo);

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    Wu = Wr;
    Wu(w1) = Wu(w1) + epsilon;
    Wu(w2) = Wu(w2) - epsilon;

    tssefo = zeros(size(X, 1), 1);

    for triali = 1 : size(X, 1) 

        yhat = forwardPass(Wi, Wu, Wo, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));
        tssefo(triali) = sum((yhat - Y(triali, :)).^2);

    end

    Ew2 = sum(tssefo);

    %%%%%%%%%%%%%%%%%%%%%%%%%%
%     Wu = Wr;
%     Wu(w1) = Wu(w1) - epsilon;
%     Wu(w2) = Wu(w2) + epsilon;
% 
%     tssefo = zeros(size(X, 1), 1);
% 
%     for triali = 1 : size(X, 1) 
% 
%         yhat = forwardPass(Wi, Wu, Wo, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));
%         tssefo(triali) = sum((yhat - Y(triali, :)).^2);
% 
%     end
% 
%     Ew3 = sum(tssefo);

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    Wu = Wr;
    Wu(w1) = Wu(w1) - epsilon;
    Wu(w2) = Wu(w2) - epsilon;

    tssefo = zeros(size(X, 1), 1);

    for triali = 1 : size(X, 1) 

        yhat = forwardPass(Wi, Wu, Wo, squeeze(X(triali, :, :)), squeeze(Y(triali, :)));
        tssefo(triali) = sum((yhat - Y(triali, :)).^2);

    end

    Ew4 = sum(tssefo);            

    H(w1) = (1/(4*(epsilon.^2)))*(Ew1 - 2*Ew2 + Ew4);
        
end

vWr = reshape(Wr, numel(Wr), 1);

weightSalience = H.*vWr;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xtrial, Ytrial, xDim] = buildTrials(trialType, xDim)

global T;


% zp = zeros(1, 5);
% op = ones(1, 5);
% 
% delay1 = zeros(1, 20);
% delay2 = zeros(1, 10);

zp = zeros(1, 10);
op = ones(1, 10);

delay1 = zeros(1, 25);
delay2 = zeros(1, 10);


if trialType == 1
    
    xDim.nConditions = 4;
    
    %%% Delayed match to sample
    %%% Match 1
    x(1, 2, :) = [zp 0 0 1 1 delay1 0 0 delay2 0 0 0 0 0 0 0];
    x(1, 3, :) = [zp 0 0 0 0 delay1 1 1 delay2 0 0 0 0 0 0 0];
    x(1, 4, :) = [zp 0 0 0 0 delay1 0 0 delay2 0 0 0 0 0 0 1];
    x(1, 5, :) = [op 0 0 0 0 delay1 0 0 delay2 0 0 0 0 0 0 1];
    x(1, 6, :) = [op 0 0 0 0 delay1 0 0 delay2 1 1 0 0 0 0 1];
    x(1, 1, :) = [ones(1, size(x, 3))];

    y(1, :)    = [zp 0 0 0 0 delay1 0 0 delay2 1 1 0 0 0 0 0];

    %%% Match 2
    x(2, 1, :) = [ones(1, size(x, 3))];
    x(2, 2, :) = [zp 0 0 -1 -1 delay1  0  0 delay2 0 0 0 0 0 0 0];
    x(2, 3, :) = [zp 0 0  0  0 delay1 -1 -1 delay2 0 0 0 0 0 0 0];
    x(2, 4, :) = [zp 0 0  0  0 delay1  0  0 delay2 0 0 0 0 0 0 1];
    x(2, 5, :) = [op 0 0  0  0 delay1  0  0 delay2 0 0 0 0 0 0 1];
    x(2, 6, :) = [op 0 0  0  0 delay1  0  0 delay2 1 1 0 0 0 0 1];

    y(2, :)    = [zp 0 0  0  0 delay1  0  0 delay2 1 1 0 0 0 0 0];

    %%% NonMatch 1
    x(3, 1, :) = [ones(1, size(x, 3))];
    x(3, 2, :) = [zp 0 0 1 1 delay1  0  0  delay2  0  0 0 0 0 0 0];
    x(3, 3, :) = [zp 0 0 0 0 delay1 -1 -1  delay2  0  0 0 0 0 0 0];
    x(3, 4, :) = [zp 0 0 0 0 delay1  0  0  delay2  0  0 0 0 0 0 1];
    x(3, 5, :) = [op 0 0 0 0 delay1  0  0  delay2  0  0 0 0 0 0 1];
    x(3, 6, :) = [op 0 0 0 0 delay1  0  0  delay2  1  1 0 0 0 0 1];

    y(3, :)    = [zp 0 0 0 0 delay1  0  0  delay2 -1 -1 0 0 0 0 0];

    %%% NonMatch 2
    x(4, 1, :) = [ones(1, size(x, 3))];
    x(4, 2, :) = [zp 0 0 -1 -1 delay1 0 0  delay2  0  0 0 0 0 0 0];
    x(4, 3, :) = [zp 0 0  0  0 delay1 1 1  delay2  0  0 0 0 0 0 0];
    x(4, 4, :) = [zp 0 0  0  0 delay1 0 0  delay2  0  0 0 0 0 0 1];
    x(4, 5, :) = [op 0 0  0  0 delay1 0 0  delay2  0  0 0 0 0 0 1];
    x(4, 6, :) = [op 0 0  0  0 delay1 0 0  delay2  1  1 0 0 0 0 1];

    y(4, :)    = [zp 0 0  0  0 delay1 0 0  delay2 -1 -1 0 0 0 0 0];    
    
else
    
    
    %%% Delayed match to sample
    %%% Match 1
    x(1, 2, :) = [zp 0 0 1 1 delay1 0 0 delay2 0 0 0 0 0 0 0];
    x(1, 3, :) = [zp 0 0 0 0 delay1 1 1 delay2 0 0 0 0 0 0 0];
    x(1, 4, :) = [zp 0 0 0 0 delay1 0 0 delay2 0 0 0 0 0 0 1];
    x(1, 5, :) = [op 0 0 0 0 delay1 0 0 delay2 0 0 0 0 0 0 1];
    x(1, 6, :) = [op 0 0 0 0 delay1 0 0 delay2 1 1 0 0 0 0 1];
    x(1, 1, :) = [ones(1, size(x, 3))];

    y(1, :)    = [zp 0 0 0 0 delay1 0 0 delay2 -1 -1 0 0 0 0 0];

    %%% Match 2
    x(2, 1, :) = [ones(1, size(x, 3))];
    x(2, 2, :) = [zp 0 0 1  1 delay1  0  0 delay2 0 0 0 0 0 0 0];
    x(2, 3, :) = [zp 0 0  0  0 delay1 -1 -1 delay2 0 0 0 0 0 0 0];
    x(2, 4, :) = [zp 0 0  0  0 delay1  0  0 delay2 0 0 0 0 0 0 1];
    x(2, 5, :) = [op 0 0  0  0 delay1  0  0 delay2 0 0 0 0 0 0 1];
    x(2, 6, :) = [op 0 0  0  0 delay1  0  0 delay2 1 1 0 0 0 0 1];

    y(2, :)    = [zp 0 0  0  0 delay1  0  0 delay2 1 1 0 0 0 0 0];

    %%% NonMatch 1
    x(3, 1, :) = [ones(1, size(x, 3))];
    x(3, 2, :) = [zp 0 0 -1 -1 delay1  0  0  delay2  0  0 0 0 0 0 0];
    x(3, 3, :) = [zp 0 0 0 0 delay1  1  1  delay2  0  0 0 0 0 0 0];
    x(3, 4, :) = [zp 0 0 0 0 delay1  0  0  delay2  0  0 0 0 0 0 1];
    x(3, 5, :) = [op 0 0 0 0 delay1  0  0  delay2  0  0 0 0 0 0 1];
    x(3, 6, :) = [op 0 0 0 0 delay1  0  0  delay2  1  1 0 0 0 0 1];

    y(3, :)    = [zp 0 0 0 0 delay1  0  0  delay2 1 1 0 0 0 0 0];

    %%% NonMatch 2
    x(4, 1, :) = [ones(1, size(x, 3))];
    x(4, 2, :) = [zp 0 0  -1 -1 delay1 0 0  delay2  0  0 0 0 0 0 0];
    x(4, 3, :) = [zp 0 0  0  0 delay1 -1 -1  delay2  0  0 0 0 0 0 0];
    x(4, 4, :) = [zp 0 0  0  0 delay1 0 0  delay2  0  0 0 0 0 0 1];
    x(4, 5, :) = [op 0 0  0  0 delay1 0 0  delay2  0  0 0 0 0 0 1];
    x(4, 6, :) = [op 0 0  0  0 delay1 0 0  delay2  1  1 0 0 0 0 1];

    y(4, :)    = [zp 0 0  0  0 delay1 0 0  delay2  -1  -1 0 0 0 0 0];   
    
    %%% Delayed match to sample
    %%% Match 1
    x(5, 2, :) = [zp 0 0  0  0 delay1 0 0 delay2 0 0 0 0 0 0 0];
    x(5, 3, :) = [zp 0 0  0  0 delay1 0 0 delay2 0 0 0 0 0 0 0];
    x(5, 4, :) = [zp 0 0  1  1 delay1 0 0 delay2 0 0 0 0 0 0 1];
    x(5, 5, :) = [op 0 0  0  0 delay1 1 1 delay2 0 0 0 0 0 0 1];
    x(5, 6, :) = [op 0 0  0  0 delay1 0 0 delay2 1 1 0 0 0 0 1];
    x(5, 1, :) = [ones(1, size(x, 3))];

    y(5, :)    = [zp 0 0 0 0 delay1 0 0 delay2  1 1 0 0 0 0 0];

    %%% Match 2
    x(6, 1, :) = [ones(1, size(x, 3))];
    x(6, 2, :) = [zp  0 0  0  0 delay1  0  0 delay2 0 0 0 0 0 0 0];
    x(6, 3, :) = [zp  0 0  0  0 delay1  0  0 delay2 0 0 0 0 0 0 0];
    x(6, 4, :) = [zp  0 0  1  1 delay1  0  0 delay2 0 0 0 0 0 0 1];
    x(6, 5, :) = [op  0 0  0  0 delay1 -1 -1 delay2 0 0 0 0 0 0 1];
    x(6, 6, :) = [op  0 0  0  0 delay1  0  0 delay2 1 1 0 0 0 0 1];

    y(6, :)    = [zp  0 0  0  0 delay1  0  0 delay2 -1 -1 0 0 0 0 0];

    %%% NonMatch 1
    x(7, 1, :) = [ones(1, size(x, 3))];
    x(7, 2, :) = [zp 0 0  0  0 delay1  0  0  delay2  0  0 0 0 0 0 0];
    x(7, 3, :) = [zp 0 0  0  0 delay1  0  0  delay2  0  0 0 0 0 0 0];
    x(7, 4, :) = [zp 0 0 -1 -1 delay1  0  0  delay2  0  0 0 0 0 0 1];
    x(7, 5, :) = [op 0 0  0  0 delay1  1  1  delay2  0  0 0 0 0 0 1];
    x(7, 6, :) = [op 0 0  0  0 delay1  0  0  delay2  1  1 0 0 0 0 1];

    y(7, :)    = [zp 0 0 0 0 delay1  0  0  delay2  -1 -1 0 0 0 0 0];

    %%% NonMatch 2
    x(8, 1, :) = [ones(1, size(x, 3))];
    x(8, 2, :) = [zp  0 0  0  0 delay1 0 0  delay2  0  0 0 0 0 0 0];
    x(8, 3, :) = [zp  0 0  0  0 delay1 0 0  delay2  0  0 0 0 0 0 0];
    x(8, 4, :) = [zp  0 0 -1 -1 delay1 0 0  delay2  0  0 0 0 0 0 1];
    x(8, 5, :) = [op  0 0  0  0 delay1 -1 -1  delay2  0  0 0 0 0 0 1];
    x(8, 6, :) = [op  0 0  0  0 delay1 0 0  delay2  1  1 0 0 0 0 1];

    y(8, :)    = [zp 0 0  0  0 delay1 0 0  delay2  1  1 0 0 0 0 0];   
    
    xDim.nConditions = size(x, 1);
    
end

Xtrial = NaN(xDim.trialRepeats, size(x, 2), size(x, 3));
Ytrial = NaN(xDim.trialRepeats, size(y, 2));

trialCtr = 1;
for repeati = 1 : xDim.trialRepeats
    for conditioni = 1 : xDim.nConditions
        condi = conditioni;
        
        Xtrial(trialCtr, :, :) = squeeze(x(condi, :, :)) + xDim.sigmax*randn(size(x, 2), size(x, 3));
        Ytrial(trialCtr, :)    = squeeze(y(condi, :))    + xDim.sigmay*randn(1, size(y, 2));
                
        trialCtr = trialCtr + 1;
        
    end
end


T = size(x, 3);
