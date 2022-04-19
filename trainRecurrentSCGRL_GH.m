function trainRecurrentSCGRL_GH(varargin)

global Li;
global Lr;
global Lo;

rng('shuffle');

xDim.stateDim     = 5;
xDim.nChoices     = 4;
xDim.nAgents      = 24;
xDim.epsilon0     = 0.15; %%% Explore/exploit parameters
xDim.epsilon      = 0.15;
xDim.epsilont     = 100;
xDim.rho          = 0.7;  %%% Learning rate for Q-learning target
xDim.beta         = 3;

xDim.recurrentEta = 0;    %%% Noise on recurrent unit activity
xDim.Qeta         = 0;    
xDim.nFixedIter   = 39;
xDim.explore      = 0;
xDim.nTrain       = 1;

paralleli = 1; %%% If running on cluster can pass this argument in

%%% Size of network
N = 20;

%%% 4 state values, choices, reward
Li = xDim.stateDim + xDim.nChoices + 1;
Lr = N;
Lo = xDim.nChoices;

fan_in = N + Li;

trainNet  = 0;   %%% Train original network
prune     = 1;   %%% Prune trained network and retrain
pruneLoad = 0;   %%% Load pruned networks for further analysis
explore   = 0;   %%% Train network on new option

%%% Prune levels   
prunevs = 0.1 : 0.2 : 0.90;  
%%% Noise level
noisev = [0.01 0.1 1];
   
%%% Train until criterion, up to 1000 iterations
fixedIter = -1;

if trainNet == 1
    
    xDim.nAgents = 24;
    
    %%% Initialize network weights    
    Wi = 0.2*randn(Lr, Li);
    Wr = sqrt(2/fan_in)*randn(Lr, Lr) + 0*eye(Lr, Lr);
    Wo = sqrt(2/N)*randn(Lo, Lr+1);
    
    [Wi, Wr, Wo, ssefo] = trainNetwork(xDim, Wi, Wr, Wo, fixedIter);

    weightName = sprintf('weightMatSCGRL_%d.mat', N);
    
    save(weightName, 'Wi', 'Wr', 'Wo', 'ssefo');
    
else
        
    weightName = sprintf('weightMatSCGRL_%d.mat', N);
    load(weightName);
    
end

if prune == 1
    
    %%% Train to a set level of iterations
    fixedIter = 1;
    
    xDim.epsilont = 10;
        
    for prunei = 1 : length(prunevs)
            
        retrainedName = sprintf('weightMatSCGRL_%d.mat', N);
        load(retrainedName);

        prunev = prunevs(prunei);
        
        %%% Load pruned and retrained or just retrained networks
        
        if prunei > 1
            
            lprunev = prunevs(prunei-1);
            prunedName    = sprintf('weightMatSCGRL_pruned_%d_%d_%d_%d.mat', N, prunei-1, 1, paralleli);
            load(prunedName);
            
            retrainedName = sprintf('weightMatSCGRL_retrained_%d_%d_%d_%d.mat', N, prunei-1, 1, paralleli);
            load(retrainedName);
            
        else
            Wipruned = Wi;
            Wrpruned = Wr;
            Wopruned = Wo;
        end
                
        [Wrpruned, clampVec] = pruneNet(xDim, Wipruned, Wrpruned, Wopruned, prunev);
        
        noisei = 2;

        xDim.Qeta = noisev(noisei);
        fprintf('Prune %.1f Eta %.2e\n', prunev, noisev(noisei));

        mnSSa       = zeros(xDim.nTrain, 1);
        mnSSapruned = zeros(xDim.nTrain, 1);     

        Wiparn = zeros(xDim.nTrain, size(Wi, 1), size(Wi, 2));
        Wrparn = zeros(xDim.nTrain, size(Wr, 1), size(Wr, 2));
        Woparn = zeros(xDim.nTrain, size(Wo, 1), size(Wo, 2));

        Wiparp = zeros(xDim.nTrain, size(Wi, 1), size(Wi, 2));
        Wrparp = zeros(xDim.nTrain, size(Wr, 1), size(Wr, 2));
        Woparp = zeros(xDim.nTrain, size(Wo, 1), size(Wo, 2));

        ssefon = zeros(xDim.nTrain, xDim.nFixedIter);
        ssefop = zeros(xDim.nTrain, xDim.nFixedIter);

        xDim.nAgents = 24;

        %%% Can train several networks in parallel on local machine
        for trainei = 1 : xDim.nTrain

            [Wio, Wro, Woo, ssefoni]                   = trainNetwork(xDim, Wi, Wr, Wo, fixedIter);
            Wiparn(trainei, :, :) = Wio;
            Wrparn(trainei, :, :) = Wro;
            Woparn(trainei, :, :) = Woo;
            ssefon(trainei, :) = ssefoni;

            [Wiprunedo, Wrprunedo, Woprunedo, ssefopi] = trainNetwork(xDim, Wipruned, Wrpruned, Wopruned, fixedIter, clampVec);
            Wiparp(trainei, :, :) = Wiprunedo;
            Wrparp(trainei, :, :) = Wrprunedo;
            Woparp(trainei, :, :) = Woprunedo;
            ssefop(trainei, :)   = ssefopi;


            [~, ~, mnSSa(trainei)]       = runQSA(xDim, Wio, Wro, Woo);  
            [~, ~, mnSSapruned(trainei)] = runQSA(xDim, Wiprunedo, Wrprunedo, Woprunedo);   

        end

        Wi = squeeze(Wiparn(1, :, :));
        Wr = squeeze(Wrparn(1, :, :));
        Wo = squeeze(Woparn(1, :, :));

        Wipruned = squeeze(Wiparp(1, :, :));
        Wrpruned = squeeze(Wrparp(1, :, :));
        Wopruned = squeeze(Woparp(1, :, :));

        fprintf('Saving file\n');

        prunedName    = sprintf('weightMatSCGRL_pruned_%d_%d_%d_%d.mat', N, prunei, noisei, paralleli);
        save(prunedName, 'Wipruned', 'Wrpruned', 'Wopruned', 'clampVec', 'ssefop', 'mnSSa');

        retrained    = sprintf('weightMatSCGRL_retrained_%d_%d_%d_%d.mat', N, prunei, noisei, paralleli);
        save(retrained, 'Wi', 'Wr', 'Wo', 'ssefon', 'mnSSapruned');

        
    end
        
elseif pruneLoad == 1
    
    loadSave = 1;
    
    Xbins = -1.5 : 0.1 : 1.5;
           
    xDim.nAgents = 64;
            
    nParallel = 100;
    
    if loadSave == 1
        
        load('RLsummary.mat');
        
        else

        mnXspruned         = nan(length(prunevs), length(noisev), nParallel);
        mnXsunpruned       = nan(length(prunevs), length(noisev), nParallel);

        mnrewPruned         = nan(length(prunevs), length(noisev), nParallel);
        mnrewUnpruned       = nan(length(prunevs), length(noisev), nParallel);

        dimPruned          = nan(length(prunevs), length(noisev), nParallel, N);
        dimUnpruned        = nan(length(prunevs), length(noisev), nParallel, N);

        rmnSSaunpruned = nan(length(prunevs), length(noisev), nParallel);
        rmnSSapruned   = nan(length(prunevs), length(noisev), nParallel);

        mnTrainingUnpruned = nan(length(prunevs), length(noisev), nParallel);
        mnTrainingPruned   = nan(length(prunevs), length(noisev), nParallel);

        vSSunpruned = nan(length(prunevs), length(noisev), nParallel);
        vSSpruned   = nan(length(prunevs), length(noisev), nParallel);

        mnLyapunovUnpruned = nan(length(prunevs), length(noisev), nParallel);
        mnLyapunovPruned   = nan(length(prunevs), length(noisev), nParallel);
        
        mnMahalUnpruned = nan(length(prunevs), length(noisev), nParallel);
        mnMahalPruned   = nan(length(prunevs), length(noisev), nParallel);
        
        mnConnectUnpruned  = nan(length(prunevs), length(noisev), nParallel, length(Xbins));
        mnConnectPruned    = nan(length(prunevs), length(noisev), nParallel, length(Xbins));

        for prunei = 1 : length(prunevs)

            for noisei = 1 : length(noisev)

                for paralleli = 1 : nParallel

                    try

                        prunedName    = sprintf('weightMatSCGRL_pruned_%d_%d_%d_%d.mat', N, prunei, noisei, paralleli);
                        load(prunedName);

                        retrained    = sprintf('weightMatSCGRL_retrained_%d_%d_%d_%d.mat', N, prunei, noisei, paralleli);
                        load(retrained);

                    catch

                        fprintf('Not available prune %d noisei %d paralleli %d\n', prunei, noisei, paralleli);
                        continue;

                    end


                    fprintf('Prune %.2f noise %.2f interaction %d\n', prunevs(prunei), noisev(noisei), paralleli);

                    [~, ~, ~, mnXsunpruned(prunei, noisei, paralleli), rewUnpruned] = runQSA(xDim, Wi, Wr, Wo);  
                    [~, ~, ~, mnXspruned(prunei, noisei, paralleli), rewPruned]     = runQSA(xDim, Wipruned, Wrpruned, Wopruned);   

                    [lyapunovu, eDistu] = estimateLyapunov(xDim, Wi, Wr, Wo);
                    [lyapunovp, eDistp] = estimateLyapunov(xDim, Wipruned, Wrpruned, Wopruned);

                    du = returnPCAD(xDim, Wi, Wr, Wo);
                    dp = returnPCAD(xDim, Wipruned, Wrpruned, Wopruned);
                    
                    [Nn, Np] = connectivityDistribution(Wr, Wrpruned);
                    
                    mnConnectUnpruned(prunei, noisei, paralleli, :) = Nn/sum(Nn);
                    mnConnectPruned(prunei, noisei, paralleli, :) = Np/sum(Np);                    

                    dimUnpruned(prunei, noisei, paralleli, :) = du/sum(du);
                    dimPruned(prunei, noisei, paralleli, :)   = dp/sum(dp);
                    
                    mnrewUnpruned(prunei, noisei, paralleli) = rewUnpruned;
                    mnrewPruned(prunei, noisei, paralleli)   = rewPruned;

                    mnTrainingUnpruned(prunei, noisei, paralleli) = nanmean(ssefon(:, end));
                    mnTrainingPruned(prunei, noisei, paralleli)   = nanmean(ssefop(:, end));

                    vSSunpruned(prunei, noisei, paralleli) = nanvar(mnSSa);
                    vSSpruned(prunei, noisei, paralleli)   = nanvar(mnSSapruned);

                    rmnSSaunpruned(prunei, noisei, paralleli) = nanmean(mnSSa, 1);
                    rmnSSapruned(prunei, noisei, paralleli)   = nanmean(mnSSapruned, 1);

                    mnLyapunovUnpruned(prunei, noisei, paralleli) = lyapunovu;
                    mnLyapunovPruned(prunei, noisei, paralleli)   = lyapunovp;
                    
                    mnMahalUnpruned(prunei, noisei, paralleli) = eDistu;
                    mnMahalPruned(prunei, noisei, paralleli) = eDistp;

                end

                fprintf('');

            end

        end
    
    end
    
%     save('RLsummary2.mat');
    
    prunePlot = 4;
    noisePlot = 2;
    
    
    mnSSnunpruned = squeeze(nanmean(nanmean(rmnSSaunpruned, 3), 2));
    mnSSnpruned   = squeeze(nanmean(nanmean(rmnSSapruned, 3), 2));

    vSSnunpruned = squeeze(nanmean(nanstd(rmnSSaunpruned, 1, 3), 2));
    vSSnpruned   = squeeze(nanmean(nanstd(rmnSSapruned, 1, 3), 2));

    mnXpunpruned = squeeze(nanmean(nanmean(mnXsunpruned(:, noisePlot, :), 3), 2));
    mnXppruned   = squeeze(nanmean(nanmean(mnXspruned(:, noisePlot, :), 3), 2));

    vXpunpruned = squeeze(nanmean(nanstd(mnXsunpruned, 1, 3), 2));
    vXppruned   = squeeze(nanmean(nanstd(mnXspruned, 1, 3), 2));

    mnTrainErrorUnpruned = squeeze(nanmean(nanmean(mnTrainingUnpruned, 3), 2));
    mnTrainErrorPruned   = squeeze(nanmean(nanmean(mnTrainingPruned, 3), 2));

    vTrainErrorUnpruned = squeeze(nanmean(nanstd(mnTrainingUnpruned, 1, 3), 2));
    vTrainErrorPruned   = squeeze(nanmean(nanstd(mnTrainingPruned, 1, 3), 2));

    mnfLyapunovUnpruned = squeeze(nanmean(nanmean(mnLyapunovUnpruned, 3), 2));
    mnfLyapunovPruned   = squeeze(nanmean(nanmean(mnLyapunovPruned, 3), 2));  

    sfLyapunovUnpruned = squeeze(nanmean(nanstd(mnLyapunovUnpruned, 1, 3), 2));
    sfLyapunovPruned   = squeeze(nanmean(nanstd(mnLyapunovPruned, 1, 3), 2));  
    
    mnfMHBUnpruned = squeeze(nanmean(nanmean(mnMahalUnpruned, 3), 2));
    mnfMHBPruned   = squeeze(nanmean(nanmean(mnMahalPruned, 3), 2));  

    sfMHBUnpruned = squeeze(nanmean(nanstd(mnMahalUnpruned, 1, 3), 2));
    sfMHBPruned   = squeeze(nanmean(nanstd(mnMahalPruned, 1, 3), 2));  
    
    mndimUnpruned      = squeeze(nanmean(dimUnpruned(prunePlot, noisePlot, :, :), 3));
    mndimPruned        = squeeze(nanmean(dimPruned(prunePlot, noisePlot, :, :), 3));
    
    rewUnpruned        = squeeze(nanmean(mnrewUnpruned(:, noisePlot, :), 3));
    rewPruned          = squeeze(nanmean(mnrewPruned(:, noisePlot, :), 3));
    
    vrewUnpruned        = squeeze(nanstd(mnrewUnpruned(:, noisePlot, :), 1, 3));
    vrewPruned          = squeeze(nanstd(mnrewPruned(:, noisePlot, :), 1, 3));
    
    connectUnpruned = squeeze(nanmean(mnConnectUnpruned(prunePlot, noisePlot, :, :), 3));
    connectPruned   = squeeze(nanmean(mnConnectPruned(prunePlot, noisePlot, :, :), 3));

    
    figure;
    subplot(3,3,1);
    errorbar([prunevs' prunevs'], [mnSSnunpruned mnSSnpruned], sqrt([vSSnunpruned vSSnpruned])/sqrt(nParallel), 'LineWidth', 2);
    axis([0 1 0 2]);
    xlabel('Prune fraction');
    ylabel('Loss');
    ax1 = gca;
    ax1.Box = 'off';
    
    [p1, pp1] = ttest(squeeze((rmnSSaunpruned(:, noisePlot, :) - rmnSSapruned(:, noisePlot, :)))');
    %%% first and last 
    [p1a, pp1a] = ttest(squeeze((rmnSSaunpruned(1, noisePlot, :) - rmnSSaunpruned(5, noisePlot, :)))');    
    
    subplot(3,3,2);    
    errorbar([prunevs' prunevs'], [mnXpunpruned mnXppruned], sqrt([vXpunpruned vXppruned])/sqrt(nParallel), 'LineWidth', 2);
    xlabel('Prune fraction');
    ylabel('Fraction correct');
    legend('Unpruned', 'Pruned');
    axis([0 1 0.8 1]);
    ax1 = gca;
    ax1.Box = 'off';
    
    [p2, pp2] = ttest(squeeze((mnXsunpruned(:, noisePlot, :) - mnXspruned(:, noisePlot, :)))');

    subplot(3,3,3);
    errorbar([prunevs' prunevs'], [rewUnpruned rewPruned], [vrewUnpruned vrewPruned]/sqrt(nParallel), 'LineWidth', 2);
    xlabel('Prune fraction');
    ylabel('Reward');
    axis([0 1 8 10.5]);
    ax1 = gca;
    ax1.Box = 'off';

    [p3, pp3] = ttest(squeeze((mnrewUnpruned(:, noisePlot, :) - mnrewPruned(:, noisePlot, :)))');
    
    subplot(3,3,4);
    errorbar([prunevs' prunevs'], [mnfLyapunovUnpruned mnfLyapunovPruned], ...
        [sfLyapunovUnpruned sfLyapunovPruned]/sqrt(nParallel), 'LineWidth', 2);
    xlabel('Prune fraction');
    ylabel('Lyapunov');
    axis([0 1 -0.06 0.03]);
    ax1 = gca;
    ax1.Box = 'off';

    [p4, pp4] = ttest(squeeze((mnLyapunovUnpruned(:, noisePlot, :) - mnLyapunovPruned(:, noisePlot, :)))');
    
    cmndimUnpruned = cumsum(mndimUnpruned);
    cmndimPruned   = cumsum(mndimPruned);
    
    dfDim = squeeze(dimUnpruned(prunePlot, noisePlot, :, :)) - squeeze(dimPruned(prunePlot, noisePlot, :, :));
    [h, p] = ttest(dfDim(:, 1:10))    

    subplot(3,3,5);
    plot(1:10, cmndimUnpruned(1:10), 1:10, cmndimPruned(1:10), 'LineWidth', 2);
    xlabel('Dimension');
    ylabel('Cumulative variance');
    ax1 = gca;
    ax1.Box = 'off';
    
    subplot(3,3,6);
    plot(Xbins, connectUnpruned, Xbins, connectPruned, 'LineWidth', 2);
    xlabel('Strength');
    ylabel('P(strength)');
    ax1 = gca;
    ax1.Box = 'off';
    
    subplot(3,3,7);
    errorbar([prunevs(1:end-1)' prunevs(1:end-1)'], [mnfMHBUnpruned(1:end-1) mnfMHBPruned(1:end-1)], ...
        [sfMHBUnpruned(1:end-1) sfMHBPruned(1:end-1)]/sqrt(nParallel), 'LineWidth', 2);
    xlabel('Prune fraction');
    ylabel('Mahalanobis');
    axis([0 1 -Inf Inf]);
    ax1 = gca;
    ax1.Box = 'off';
    
    [p5, pp5] = ttest(squeeze((mnMahalPruned(:, noisePlot, :) - mnMahalUnpruned(:, noisePlot, :)))');
    
    for noisevi = 1 : 3

        mnLb = [squeeze(mnLyapunovUnpruned(prunePlot, noisevi, :)); squeeze(mnLyapunovPruned(prunePlot, noisevi, :))];
        mnLu = [squeeze(mnLyapunovUnpruned(prunePlot, noisevi, :))];
        mnLp = [squeeze(mnLyapunovPruned(prunePlot, noisevi, :))];

        mnCb = [squeeze(mnXsunpruned(prunePlot, noisevi, :)); squeeze(mnXspruned(prunePlot, noisevi, :))];
        mnCu = [squeeze(mnXsunpruned(prunePlot, noisevi, :))];
        mnCp = [squeeze(mnXspruned(prunePlot, noisevi, :))];

        [crvb, pvb] = corrcoef([mnLb mnCb]);           
        [crvu, pvu] = corrcoef([mnLu mnCu]);           
        [crvp, pvp] = corrcoef([mnLp mnCp]);           

        fprintf('Correlation between lyapunov and fraction correct both %.3f p %.3f\n', crvb(1,2), pvb(1,2));
        fprintf('Correlation between lyapunov and fraction correct Unpruned %.3f p %.3f\n', crvu(1,2), pvu(1,2));
        fprintf('Correlation between lyapunov and fraction correct Pruned %.3f p %.3f\n', crvp(1,2), pvp(1,2));

        subplot(3,3, 6+noisevi);
        plot(squeeze(mnXsunpruned(prunePlot, noisevi, :)), squeeze(mnLyapunovUnpruned(prunePlot, noisevi, :)), '.', ...
             squeeze(mnXspruned(prunePlot, noisevi, :)), squeeze(mnLyapunovPruned(prunePlot, noisevi, :)), '.', 'MarkerSize', 20);
        xlabel('Fraction correct');
        ylabel('Mean Lyapunov');
        if noisevi == 1
            axis([0.955 0.99 -Inf Inf]);
        end
        ax1 = gca;
        ax1.Box = 'off';

    end 
        
%     subplot(2,2,3);
%     errorbar([prunevs' prunevs'], [mnTrainErrorUnpruned mnTrainErrorPruned], ...
%         [vTrainErrorUnpruned vTrainErrorPruned]/sqrt(nParallel));
%     xlabel('Prune fraction');
%     ylabel('Training Loss');
%     legend('unpruned', 'Pruned');
%     axis([0 1 1.2 1.5]);
%     
%     subplot(2,2,4);
%     plot((1:size(ssefon, 2))', ssefon(1, :)', (1:size(ssefop, 2))', ssefop(1, :)');
%     xlabel('Training iteration');
%     ylabel('Loss');
    
end

%%% Example plot for trained pruned and unpruned networks
if probeTraj == 1
    
    prunei = 7;
    noisei = find(noisev == 1);

    xDim.beta = 3;
    xDim.Qeta = noisev(noisei);
    xDim.epsilon = 0;
    
    xDim.nAgents = 128;
    
    filei = 4;

    prunedName    = sprintf('weightMatSCGRL_pruned_%d_%d_%d_%d.mat', N, 4, 2, filei);
    load(prunedName);

    retrained    = sprintf('weightMatSCGRL_retrained_%d_%d_%d_%d.mat', N, 4, 2, filei);
    load(retrained);

    runQSA(xDim, Wi, Wr, Wo, 1);  
    runQSA(xDim, Wipruned, Wrpruned, Wopruned, 1);   

    du = dimensionalityReduction(xDim, Wi, Wr, Wo, 1);
    dp = dimensionalityReduction(xDim, Wipruned, Wrpruned, Wopruned, 1);

    connectivityDistribution(Wr, Wrpruned);
    subplot(2,2,4);
    plot(1:length(dp), cumsum(du)/sum(du), 1:length(dp), cumsum(dp)/sum(dp));
    legend('Unpruned', 'Pruned');

    [lyapunovu, eDistu] = estimateLyapunov(xDim, Wi, Wr, Wo);
    [lyapunovp, eDistp] = estimateLyapunov(xDim, Wipruned, Wrpruned, Wopruned);

    fprintf('Unpruned L %.3f MH %.3f pruned L %.3f MH %.3f\n', mean(lyapunovu), mean(eDistu), mean(lyapunovp), mean(eDistp));

    fprintf('');
        
    
end

if explore == 1
    
    fixedIter = 1;
       
    xDim.explore = 1;
    
    prunedName    = sprintf('weightMatSCGRL_pruned_%d_%d_%d_%d.mat', N, 4, 2, paralleli);
    load(prunedName);

    z = (Wrpruned == 0);
    WrclampVec = ones(numel(Wrpruned), 1);
    WrclampVec(z) = 0;

    totalElements = numel(Wi) + numel(Wr) + numel(Wo);
    clampVec = ones(totalElements, 1);
    clampVec(numel(Wi)+1:numel(Wi) + numel(Wr)) = WrclampVec;

    oneClamp = ones(totalElements, 1);

    retrained    = sprintf('weightMatSCGRL_retrained_%d_%d_%d_%d.mat', N, 4, 2, paralleli);
    load(retrained);

    [~, ~, ~, sseu] = trainNetwork(xDim, Wi, Wr, Wo, fixedIter, oneClamp, 40);
    [~, ~, ~, ssep] = trainNetwork(xDim, Wipruned, Wrpruned, Wopruned, fixedIter, clampVec, 40);
    
    retrainExplore  = sprintf('weightMatSCGRL_retrainExp_%d_%d_%d_%d.mat', N, 4, 2, paralleli);
    prunedExplore   = sprintf('weightMatSCGRL_prunedExp_%d_%d_%d_%d.mat', N, 4, 2, paralleli);
    
    save(prunedExplore, 'ssep');
    save(retrainExplore, 'sseu');
    
end

if explorePlot == 1
    
    nfiles = 100;
    
    ssefou = zeros(nfiles, xDim.nFixedIter);
    ssefop = zeros(nfiles, xDim.nFixedIter);

    for paralleli = 1 : nfiles
                   
        prunedName    = sprintf('weightMatSCGRL_prunedExp_%d_%d_%d_%d.mat', N, 4, 2, paralleli);
        load(prunedName);

        retrained    = sprintf('weightMatSCGRL_retrainExp_%d_%d_%d_%d.mat', N, 4, 2, paralleli);
        load(retrained);
        
        ssefou(paralleli, :) = sseu;
        ssefop(paralleli, :) = ssep;

    end
    
    figure;
    
    z = find(ssefou(:, 10) < 0.1);
    
    zp = find(ssefou(:, 2) < 0.1);
    
    mnssefou = median(ssefou);
    mnssefop = median(ssefop);
    
    ssssefou = [std(ssefou(zp, 1:4))/sqrt(length(z)) std(ssefou(z, 5:end))/sqrt(length(z))];
    ssssefop = std(ssefop)/sqrt(nfiles);
    
    trainIterations = 2:xDim.nFixedIter;
    
     errorbar([trainIterations' trainIterations'], [mnssefou(2:end)' mnssefop(2:end)'], [ssssefou(2:end)' ssssefop(2:end)']);
     axis([1 xDim.nFixedIter 0 0.05]);
    
    plot([trainIterations' trainIterations'], [mnssefou(2:end)' mnssefop(2:end)']);
    legend('Unpruned', 'Pruned');
    
end


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

Xbins = -1.5 : 0.1 : 1.5;
[Nn, X] = hist(vWrn(zin), Xbins);

[Np, X] = hist(vWrp(zip), Xbins);

figure;
subplot(2,2,1);
plot(1:nDims, cumsum(diag(Sn))/sum(diag(Sn)), 1:nDims, cumsum(diag(Sp))/sum(diag(Sp)));
legend('Unpruned', 'Pruned');

subplot(2,2,2);
plot(X, Nn/sum(Nn), X, Np/sum(Np));

subplot(2,2,3);
plot(1:length(sn), sn, 1:length(sp), sp);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lyapunov, eDist] = estimateLyapunov(xDim, Wi, Wr, Wo)

xDim.nAgents = 48;

[X, Y] = runQSA(xDim, Wi, Wr, Wo);

T = size(X, 3);

nDims = size(Wr, 1);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%PCA axes
Aft = NaN(nDims, T*xDim.nAgents);

AftCondition = NaN(xDim.nAgents, nDims, T);

for agenti = 1 : xDim.nAgents
    
    [~, a] = forwardPass(xDim, Wi, Wr, Wo, squeeze(X(agenti, :, :)));
    
    sbin = (agenti-1)*T + 1;
    ebin = sbin + T - 1;
    Aft(:, sbin:ebin) = a;
    
    AftCondition(agenti, :, :) = a;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Compute distance betweeen centroids
    mnCh1 = mean(squeeze(AftCondition(agenti, :, 1:2:end)), 2);
    mnCh2 = mean(squeeze(AftCondition(agenti, :, 2:2:end)), 2);

    cv1 = cov(squeeze(AftCondition(agenti, :, 1:2:end))');
    cv2 = cov(squeeze(AftCondition(agenti, :, 2:2:end))');

    mncv = (cv1 + cv2)/2;

    dfMean = mnCh1 - mnCh2;

    eDist(agenti) = dfMean'*inv(mncv)*dfMean;

%     fprintf('Diff in full D means %.4f\n', eDist);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Jacobian
    tOn  = 2;
    tOff = T;

    for ti = tOn : tOff

        J = computeJacob(Wr, AftCondition, ti, agenti);

        if ti == tOn
            JT = J;
        else
            JT = JT*J;
        end

    end

    JT = JT'*JT;

    [~, DJ] = eig(JT);

    lyapunov(agenti) = max(log(abs(diag(DJ))))/(2*(T-tOn));

%     fprintf('Max lyapunov %.3f\n', log(max(lyapunov)));
    
end

lyapunov = mean(lyapunov);
eDist = mean(eDist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wr, clampVec] = pruneNet(xDim, Wi, Wr, Wo, prunev)

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

%%% Unused methods for selecting weights to prune
% [X, Y] = runQSA(xDim, Wi, Wr, Wo);
% [~, dWr, ~] = backProp(xDim, Wi, Wr, Wo, X, Y);
% 
% vdWr = reshape(dWr, numel(dWr), 1);
vWr  = reshape(Wr, numel(Wr), 1);

% v2dWr = vdWr.^2;
v2Wr  = vWr.^2;

% HvweightSalience = v2dWr.*v2Wr;

HvweightSalience = v2Wr;

[~, w_index] = sort(HvweightSalience, 'ascend');

Wr(w_index(1:nPrune)) = 0;

WrclampVec = ones(numel(Wr), 1);
WrclampVec(w_index(1:nPrune)) = 0;

totalElements = numel(Wi) + numel(Wr) + numel(Wo);
clampVec = ones(totalElements, 1);
clampVec(numel(Wi)+1:numel(Wi) + numel(Wr)) = WrclampVec;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d] = returnPCAD(xDim, Wi, Wr, Wo)

nDims = size(Wr, 1);

[X, ~] = runQSA(xDim, Wi, Wr, Wo);

T = size(X, 3);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%PCA axes
Aft = NaN(nDims, T*xDim.nAgents);

AftCondition = NaN(xDim.nAgents, nDims, T);

for agenti = 1 : xDim.nAgents
    
    [~, a] = forwardPass(xDim, Wi, Wr, Wo, squeeze(X(agenti, :, :)));
    
    sbin = (agenti-1)*T + 1;
    ebin = sbin + T - 1;
    Aft(:, sbin:ebin) = a;
    
    AftCondition(agenti, :, :) = a;
            
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PCA space
cvA = cov(Aft');

[V, D] = eig(cvA);

d = diag(D);

d = d(end:-1:1);
V = V(:, end:-1:1);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d] = dimensionalityReduction(xDim, Wi, Wr, Wo, plotSubspace)

nDims = size(Wr, 1);

[X, Y] = runQSA(xDim, Wi, Wr, Wo, 1);

T = size(X, 3);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%PCA axes
Aft = NaN(nDims, T*xDim.nAgents);

AftCondition = NaN(xDim.nAgents, nDims, T);

for agenti = 1 : xDim.nAgents
    
    [~, a] = forwardPass(xDim, Wi, Wr, Wo, squeeze(X(agenti, :, :)));
    
    sbin = (agenti-1)*T + 1;
    ebin = sbin + T - 1;
    Aft(:, sbin:ebin) = a;
    
    AftCondition(agenti, :, :) = a;
            
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Compute distance betweeen centroids
mnCh1 = mean(squeeze(AftCondition(1, :, 1:2:end)), 2);
mnCh2 = mean(squeeze(AftCondition(1, :, 2:2:end)), 2);

cv1 = cov(squeeze(AftCondition(1, :, 1:2:end))');
cv2 = cov(squeeze(AftCondition(1, :, 2:2:end))');

mncv = (cv1 + cv2)/2;

dfMean = mnCh1 - mnCh2;

eDist = dfMean'*inv(mncv)*dfMean;

fprintf('Diff in full D means %.4f\n', eDist);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PCA space
cvA = cov(Aft');

[V, D] = eig(cvA);

d = diag(D);

d = d(end:-1:1);
V = V(:, end:-1:1);

nProj = 3;

Vp = V(:, 1:nProj);

pMat = NaN(xDim.nAgents, nProj, T);

spd = NaN(xDim.nAgents, T-1);

ypred = NaN(xDim.nAgents, xDim.nChoices, T);

for agenti = 1 : 1
    
    [y, a] = forwardPass(xDim, Wi, Wr, Wo, squeeze(X(agenti, :, :)));
    
    spd(agenti, :) = sum(abs(diff(a')).^2, 2);
    
    pMat(agenti, :, :) = Vp'*a;    
    
    ypred(agenti, :, :) = y;
    
end

mnCh1 = mean(squeeze(pMat(1, :, 1:2:end)), 2);
mnCh2 = mean(squeeze(pMat(1, :, 2:2:end)), 2);

cv1 = cov(squeeze(pMat(1, :, 1:2:end))');
cv2 = cov(squeeze(pMat(1, :, 2:2:end))');

mncv = (cv1 + cv2)/2;

dfMean = mnCh1 - mnCh2;

eDist = dfMean'*inv(mncv)*dfMean;

fprintf('Diff in PCA means %.4f\n', eDist);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Jacobian
tOn  = 2;
tOff = T;

for ti = tOn : tOff
    
    J = computeJacob(Wr, AftCondition, ti, 1);
            
    if ti == tOn
        JT = J;
    else
        JT = JT*J;
    end
        
end

JT = JT'*JT;

[~, DJ] = eig(JT);

lyapunov = diag(DJ);

fprintf('Max lyapunov %.3f\n', log(max(lyapunov))/(2*(T-2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% corr([Wo(:, 1:end-1)' Vp])

x3 = 0;

% z = find(Y(1, 3, 2:2:end) > 0.8);
z = 8;

lptc = z(1)-2;
lptp = z(1)-1;

afp1 = a(:, lptc);
afp2 = a(:, lptc+1);

Xi = squeeze(X(1, :, :));

Xir1 = [Xi(1:xDim.stateDim, lptp)' 0 0 1 0 1]';
Xin1 = [Xi(1:xDim.stateDim, lptp)' 0 0 1 0 0]';

Xir2 = [Xi(1:xDim.stateDim, lptp+1)' 1 0 0 0 0]';
Xin2 = [Xi(1:xDim.stateDim, lptp+1)' 1 0 0 0 -1]';

pctr = 0;
for x1 = 0 : 0.5 : 0
    for x2 = 0 : 0.5 : 0

        pctr = pctr + 1;

        dx1 = afp1 + Vp*[x1; x2; x3];
        dx2 = afp2 + Vp*[x1; x2; x3];

        da1 =  Wi*Xir1 + Wr*tanh(dx1) - dx1;
        da2 =  Wi*Xir2 + Wr*tanh(dx2) - dx2;
        
        spfp(pctr) = sum(abs(da1).^2);
        
        nxi(pctr) = norm([x1; x2]);

        dap1(pctr, :) = Vp'*da1;
        dxp1(pctr, :) = Vp'*dx1;
        dap2(pctr, :) = Vp'*da2;
        dxp2(pctr, :) = Vp'*dx2;
            
    end
end

corr([squeeze(Y(1, 1:4, :))' squeeze(pMat(1, :, :))']);

if plotSubspace == 1
    
    figure;
    hold on;
    for choicei = 1 : 2
        plot(squeeze(pMat(1, 1, choicei:2:end))', squeeze(pMat(1, 2, choicei:2:end))', ...
             'LineWidth', 2, 'Marker', 'o');    

        text(squeeze(pMat(1, 1, choicei)),  squeeze(pMat(1, 2, choicei)),'s', 'Fontsize', 20);

%         if choicei == 1
%             lpt = lptp;
%         else
%             lpt = lptc;
%         end
%         text(squeeze(pMat(1, 1, lpt)),  squeeze(pMat(1, 2, lpt)),  squeeze(pMat(1, 3, lpt)),'L', 'Fontsize', 20);
    end
    
    xlabel('PCA1');
    ylabel('PCA2');

    for pti = 1 : pctr

        lx = [dxp1(pti, 1) (dxp1(pti, 1) + dap1(pti, 1))];
        ly = [dxp1(pti, 2) (dxp1(pti, 2) + dap1(pti, 2))];

        line(lx, ly);
        text(dxp1(pti, 1), dxp1(pti, 2), 'o', ...
            'HorizontalAlignment', 'center');

        lx = [dxp2(pti, 1) (dxp2(pti, 1) + dap2(pti, 1))];
        ly = [dxp2(pti, 2) (dxp2(pti, 2) + dap2(pti, 2))];

        line(lx, ly);
        text(dxp2(pti, 1), dxp2(pti, 2), 'o', ...
            'HorizontalAlignment', 'center');

    end

    for ti = 2 : 2 : T-1

        trg = Xi(xDim.stateDim+3, ti+1) == 1;
        rew = Xi(end, ti+1);

        if trg == 1
            if rew == 1
                text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),'+', 'Fontsize', 20, 'HorizontalAlignment', 'center');
            else
                text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),'-', 'Fontsize', 20, 'HorizontalAlignment', 'center');
            end
        else
            if rew == 1
                text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),'+', 'Fontsize', 20,'Color','red', 'HorizontalAlignment', 'center');
            else
                text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),'-', 'Fontsize', 20,'Color','red', 'HorizontalAlignment', 'center');
            end
        end

    end

    view(0, 90);

%     figure
%     subplot(2,2,1);
%     plot(1:T/2, squeeze(ypred(1, :, 2:2:end)));
% 
%     subplot(2,2,2);
%     plot(1:T/2, squeeze(ypred(1, :, 1:2:end)));

    fprintf('');
    
end

%%% 3D
% if plotSubspace == 1
%     
%     figure;
%     hold on;
%     for choicei = 1 : 2
%         plot3(squeeze(pMat(1, 1, choicei:2:end))', squeeze(pMat(1, 2, choicei:2:end))', ...
%               squeeze(pMat(1, 3, choicei:2:end))', 'LineWidth', 2, 'Marker', 'o');    
% 
%         text(squeeze(pMat(1, 1, choicei)),  squeeze(pMat(1, 2, choicei)),  squeeze(pMat(1, 3, choicei)),'s', 'Fontsize', 20);
% 
%         if choicei == 1
%             lpt = lptp;
%         else
%             lpt = lptc;
%         end
%         text(squeeze(pMat(1, 1, lpt)),  squeeze(pMat(1, 2, lpt)),  squeeze(pMat(1, 3, lpt)),'L', 'Fontsize', 20);
%     end
%     
%     xlabel('PCA1');
%     ylabel('PCA2');
%     zlabel('PCA3');
% 
%     for pti = 1 : pctr
% 
%         lx = [dxp1(pti, 1) (dxp1(pti, 1) + dap1(pti, 1))];
%         ly = [dxp1(pti, 2) (dxp1(pti, 2) + dap1(pti, 2))];
%         lz = [dxp1(pti, 3) (dxp1(pti, 3) + dap1(pti, 3))];
% 
%         line(lx, ly, lz);
%         text(dxp1(pti, 1), dxp1(pti, 2), dxp1(pti, 3), 'o', ...
%             'HorizontalAlignment', 'center');
% 
%         lx = [dxp2(pti, 1) (dxp2(pti, 1) + dap2(pti, 1))];
%         ly = [dxp2(pti, 2) (dxp2(pti, 2) + dap2(pti, 2))];
%         lz = [dxp2(pti, 3) (dxp2(pti, 3) + dap2(pti, 3))];
% 
%         line(lx, ly, lz);
%         text(dxp2(pti, 1), dxp2(pti, 2), dxp2(pti, 3), 'o', ...
%             'HorizontalAlignment', 'center');
% 
%     end
% 
%     for ti = 2 : 2 : T-1
% 
%         trg = Xi(xDim.stateDim+3, ti+1) == 1;
%         rew = Xi(end, ti+1);
% 
%         if trg == 1
%             if rew == 1
%                 text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),  squeeze(pMat(1, 3, ti)),'+', 'Fontsize', 20, 'HorizontalAlignment', 'center');
%             else
%                 text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),  squeeze(pMat(1, 3, ti)),'-', 'Fontsize', 20, 'HorizontalAlignment', 'center');
%             end
%         else
%             if rew == 1
%                 text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),  squeeze(pMat(1, 3, ti)),'+', 'Fontsize', 20,'Color','red', 'HorizontalAlignment', 'center');
%             else
%                 text(squeeze(pMat(1, 1, ti)),  squeeze(pMat(1, 2, ti)),  squeeze(pMat(1, 3, ti)),'-', 'Fontsize', 20,'Color','red', 'HorizontalAlignment', 'center');
%             end
%         end
% 
%     end
% 
%     view(0, 90);
% 
% 
% 
% %     figure
% %     subplot(2,2,1);
% %     plot(1:T/2, squeeze(ypred(1, :, 2:2:end)));
% % 
% %     subplot(2,2,2);
% %     plot(1:T/2, squeeze(ypred(1, :, 1:2:end)));
% 
%     fprintf('');
%     
% end

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
function [Wi, Wr, Wo, ssefo] = trainNetwork(xDim, Wi, Wr, Wo, fixedIter, varargin)
   
if ~isempty(varargin)
    clampVec = varargin{1};
else
    totalElements = numel(Wi) + numel(Wr) + numel(Wo);
    clampVec = ones(totalElements, 1);    
end

if size(varargin, 2) > 1
    trainLength = varargin{2};
else
    trainLength = 20;
end

if fixedIter == -1

    maxBPIter = 1000;

else

    %%% Note this should not be a multiple of 20
    maxBPIter = xDim.nFixedIter;

end

ssefo = NaN(maxBPIter, 1);

[X, Y] = runQSA(xDim, Wi, Wr, Wo);

for iter = 1 : maxBPIter
    
    if iter < xDim.epsilont
        xDim.epsilon = xDim.epsilon0*(1 - iter/xDim.epsilont);
    else
        xDim.epsilon = 0;
    end
    
    if mod(iter, trainLength) == 0
        [X, Y] = runQSA(xDim, Wi, Wr, Wo);
    end

    [dWi, dWr, dWo, sseft] = backProp(xDim, Wi, Wr, Wo, X, Y);

    ssefo(iter) = sseft;   

    dEdtheta = vecWeights(dWi, dWr, dWo);

    [Wi, Wr, Wo, converged] = conjGrad(xDim, dEdtheta, Wi, Wr, Wo, X, Y, sseft, clampVec); 

    if converged == 1
        break;
    end

end

% plot(ssefo(10:end));
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, Y, mnSSa, mnxAgent, mnRew] = runQSA(xDim, Wi, Wr, Wo, varargin)

if size(varargin, 2) > 0
    plotTraj = varargin{1};
else
    plotTraj = 0;
end

% plotTraj = 1;

X = buildTrials(xDim);

T = size(X, 3);

Y = zeros(xDim.nAgents, xDim.nChoices, T);

tr = T;

pv = [0.8 0.2 0];

for agenti = 1 : xDim.nAgents    
    
    if xDim.explore == 0
    
        if rem(agenti, 2) == 0
            p1 = pv(1);
            p2 = 1 - p1;
            p3 = 0;
        else
            p1 = pv(2);
            p2 = 1 - p1;
            p3 = 0;
        end

        if p1 > p2
            bestp = 2;
        else
            bestp = 3;
        end

        rewardProb(:, 1) = [-1; p1; p2; p3];
        rewardProb(:, 2) = [-1; p2; p1; p3];
        
    else
        if rem(agenti, 2) == 0
            p1 = pv(1);
            p2 = 0;
            p3 = 1-p1;
        else
            p1 = pv(2);
            p2 = 0;
            p3 = 1-p1;
        end

        if p1 > p3
            bestp = 2;
        else
            bestp = 4;
        end

        rewardProb(:, 1) = [-1; p1; p2; p3];
        rewardProb(:, 2) = [-1; p2; p1; p3];
    end
    
    Xi = squeeze(X(agenti, :, :));
    
    Q      = 0.5*ones(4, xDim.nChoices, T+1);
    rew    = zeros(T, 1);
    choice = zeros(T, 1);
    dTrial = zeros(xDim.nChoices, T);
    qchoice = zeros(T, 1);
    choiceRand = rand(T, 1);
    
    rewardRand = rand(T, 1);
    
    correct      = 0;
    qcorrect     = 0;
    crossCorrect = 0;
    
    sse = 0;
    
    eta = sqrt(xDim.Qeta)*randn(xDim.nChoices, T);

    for ti = 1 : T
        
        yhat = forwardPass(xDim, Wi, Wr, Wo, Xi(:, 1:ti));
        
        if Xi(2, ti) == 1
            sti = 1;
        elseif Xi(3, ti) == 1 && Xi(4, ti) == 1 && Xi(5, ti) == 0
            sti = 2;
        elseif Xi(3, ti) == 1 && Xi(4, ti) == 0 && Xi(5, ti) == 1
            sti = 3;
        elseif Xi(3, ti) == 0 && Xi(4, ti) == 1 && Xi(5, ti) == 1
            sti = 4;
        else
            fprintf('bad state\n');
        end
        
        Qd = exp(xDim.beta*squeeze(Q(sti, :, ti)))/sum(exp(xDim.beta*squeeze(Q(sti, :, ti))));

        %%% Convert choice output to choice probability
        cv = yhat(:, ti);
        d  = exp(xDim.beta*cv)./sum(exp(xDim.beta*cv));

        %%% explore?
        if rand(1,1) < xDim.epsilon
            d = ones(xDim.nChoices, 1)*1/length(d);
            Qd = ones(xDim.nChoices, 1)*1/length(d);
        end

        spc  = cumsum(d);
        Qspc = cumsum(Qd);

        choicev = find((spc - choiceRand(ti)) >= 0);
        qchoicev = find((Qspc - choiceRand(ti)) >= 0);
        
        choice(ti) = choicev(1);
        qchoice(ti) = qchoicev(1);
        
        dTrial(:, ti) = d;
        
        if choice(ti) == qchoice(ti)
            crossCorrect = crossCorrect + 1;
        end
        
        if Xi(3, ti) == 1
            
            if choice(ti) == bestp            
                correct = correct + 1;
            end
            
            if qchoice(ti) == bestp
                qcorrect = qcorrect + 1;
            end
            
        elseif Xi(2, ti) == 1
                
            if choice(ti) == 1
                correct = correct + 1;
            end
            
            if qchoice(ti) == 1
                qcorrect = qcorrect + 1;
            end
            
        end

        rew(ti) = taskReward(Xi, choice, rewardRand, rewardProb, ti, tr);
        
        Q(:, :, ti+1) = Q(:, :, ti);
        
        Q(sti, choice(ti), ti+1) = Q(sti, choice(ti), ti) + xDim.rho*(rew(ti) - Q(sti, choice(ti), ti));
        
        Y(agenti, :, ti) = Q(sti, :, ti) + eta(ti);
        
        %%% Feed choice and reward to next time point
        if ti < T
            
            X(agenti, xDim.stateDim+choice(ti), ti+1) = 1;
            X(agenti, end, ti+1) = rew(ti);
            
            Xi(xDim.stateDim+choice(ti), ti+1) = 1;
            Xi(end, ti+1) = rew(ti);
            
        end
        
        sse = sse + sum((cv - squeeze(Q(sti, :, ti))').^2);
        
    end
    
    ssAgent(agenti) = sse;
    
    fcAgent(agenti) = correct/T;
    qAgent(agenti) = qcorrect/T;
    
    xAgent(agenti) = crossCorrect/T;
    
    rewAgent(agenti) = sum(rew);
    
    if plotTraj == 1 && agenti == 1
    
        figure;
        subplot(2,2,1);
        plot(1:(size(yhat, 2)/2), yhat(1:xDim.nChoices, 2:2:end), 'LineWidth', 2);
        hold on;
        plot(1:(T/2), squeeze(Q(2, 1:xDim.nChoices, 2:2:T)), '-.', 'LineWidth', 2);
        legend('y0', 'y1', 'y2', 'y3', 'q0', 'q1', 'q2', 'q4');

        subplot(2,2,2);
        plot(1:(size(yhat, 2)/2), yhat(1:xDim.nChoices, 1:2:end), 'LineWidth', 2);
        hold on;
        plot(1:(T/2), squeeze(Q(1, 1:xDim.nChoices, 1:2:T)), '-.', 'LineWidth', 2);
        
        subplot(2,2,3);
        plot(1:(size(yhat, 2)/2), dTrial(:, 2:2:end), 'LineWidth', 2);
        
    end

    fprintf('');
    
end

mnSSa = mean(ssAgent)/(size(X, 3));

mnxAgent = mean(xAgent);

mnRew = mean(rewAgent);

fprintf('Total Correct %.3f Q %.3f Cross %.3f sse %.3f rew %.2f\n', mean(fcAgent), mean(qAgent), mean(xAgent), mnSSa, mnRew);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rew = taskReward(Xi, choice, rewardRand, rewardProb, t, tr)

%%% Fixation is target 1
if Xi(2, t) == 1
    if choice(t) ~= 1
        rew = -1;
    else
        rew = 0;
    end
%%% Target 2 and 3 on
elseif Xi(3, t) == 1 && Xi(4, t) == 1
    if choice(t) == 1 || choice(t) == 4
        rew = 0;
    else
        if t <= tr
            rew = double(rewardRand(t) < rewardProb(choice(t), 1));
        else
            rew = double(rewardRand(t) < rewardProb(choice(t), 2));
        end
    end
elseif Xi(3, t) == 1 && Xi(4, t) == 0 && Xi(5, t) == 1
    if choice(t) == 1 
        rew = 0;
    else
        if t <= tr
            rew = double(rewardRand(t) < rewardProb(choice(t), 1));
        else
            rew = double(rewardRand(t) < rewardProb(choice(t), 2));
        end
    end
    
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wi, Wr, Wo, converged] = conjGrad(xDim, dEdtheta, Wi, Wr, Wo, X, Y, ssef0, clampVec)

vWk = vecWeights(Wi, Wr, Wo);
    
matSize = [size(Wi, 2) size(Wr, 1) size(Wo, 1)];

maxCGIter = numel(vWk);

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

for cgIter = 1 : maxCGIter

    npk = norm(pk);

    if success == 1
        
        sigmak = sigma/npk;

        vWkd = vWk + sigmak*pk;

        [Wik, Wrk, Wok] = matWeights(vWkd, matSize);

        [dWik, dWrk, dWok] = backProp(xDim, Wik, Wrk, Wok, X, Y);

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
        
    [dWi, dWr, dWo, ssef] = backProp(xDim, tWi, tWr, tWo, X, Y);
    
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
            fprintf('CG %d sse %.4f delta %.3e Delta %.3e lambda %.3e alpha %.4f success %d\n', ...
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
        dfEk = 0.001*(Ekm1 - Ek) + 0.999*dfEk;     
    else
        dfEk = Inf;
    end
    
    if Ek < 0.01 || (Ek < 0.015 && dfEk < 0.00000001 && cgIter > 20 && success == 1)
%     if dfEk < 0.00000001 && cgIter > 20 && success == 1
%     if Ek < 10^-8 && success == 1

        fprintf('Cgiter %d Ek %.5f Ek-1 %.5f %.5f\n', cgIter, Ek, Ekm1, dfEk);

        [Wi, Wr, Wo] = matWeights(vWk);
        converged = 1;
        return;
        
    end
        
    if rem(cgIter, 100) == 0
        fprintf('CG %d sse %.8f delta %.3e Delta %.3f lambda %.3e alpha %.4f success %d\n', ...
            cgIter, Ek, deltak, gDeltak, lambdak, alphak, success);
    end
    
end

[Wi, Wr, Wo] = matWeights(vWk);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dWi, dWr, dWo, ssef] = backProp(xDim, Wi, Wr, Wo, X, Y)

tdWi = zeros(size(X, 1), size(Wi, 1), size(Wi, 2));
tdWr = zeros(size(X, 1), size(Wr, 1), size(Wr, 2));
tdWo = zeros(size(X, 1), size(Wo, 1), size(Wo, 2));

tssefo = zeros(size(X, 1), 1);

for agenti = 1 : size(X, 1) 
        
    [yhat, a, z] = forwardPass(xDim, Wi, Wr, Wo, squeeze(X(agenti, :, :)));

    [dWi, dWr, dWo] = backWardPass(Wr, Wo, yhat, a, z, squeeze(X(agenti, :, :)), squeeze(Y(agenti, :, :)));

    if sum((size(dWo) - size(Wo)).^2) ~= 0
        dWo = dWo';
    end

    tdWi(agenti, :, :) = dWi;
    tdWr(agenti, :, :) = dWr;
    tdWo(agenti, :, :) = dWo;

    tssefo(agenti) = sum(sum((yhat - squeeze(Y(agenti, :, :))).^2));

end

nPoints = size(X, 1)*size(X, 3);

dWi = squeeze(sum(tdWi, 1))/nPoints;
dWr = squeeze(sum(tdWr, 1))/nPoints;
dWo = squeeze(sum(tdWo, 1))/nPoints;

ssef = sum(tssefo)/nPoints;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [yhat, a, z] = forwardPass(xDim, Wi, Wr, Wo, Xi)

T = size(Xi, 2);
Lr = size(Wr, 1);
Lo = size(Wo, 1);

yhat = NaN(Lo, T);

a    = zeros(Lr, T);
z    = zeros(Lr, T);
z0   = zeros(Lr, 1);
z(:, 1) = z0;

eta = sqrt(xDim.recurrentEta)*randn(Lr, T);

for t = 1 : T

    %%% Forward pass
    %%% Linear term
    if t > 1
        a(:, t) = Wi*Xi(:, t) + Wr*z(:, t-1);
    else
        a(:, t) = Wi*Xi(:, t);
    end

    %%% Nonlinear transfer
    z(:,t) = tanh(a(:,t) + eta(:, t));

    %%% Output
    yhat(:, t) = Wo*[z(:,t); 1];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dWi, dWr, dWo] = backWardPass(Wr, Wo, yhat, a, z, Xi, y)


T = size(y, 2);
Lr = size(Wr, 1);
Lo = size(Wo, 1);
Li = size(Xi, 1);

Wonb = Wo(:, 1:Lr);

drt   = zeros(Lr, T+1);

dedwo = zeros(T, Lo, Lr+1);
dedwi = zeros(T, Lr, Li);
dedwr = zeros(T, Lr, Lr); 

dy = 2*(yhat - y);

for t = length(y) : -1 : 1

    %%% Backward pass        
    dft = dftanh(a(:,t));
    
    drt(:, t) = dft.*((Wonb'*dy(:,t)) + (Wr'*drt(:,t+1)));

    dedwo(t, :, :) = dy(:,t)*[z(:,t); 1]';
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
mWo = reshape(vWo, Lo, Lr+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dh = dftanh(x)

dh = 1 - tanh(x).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X] = buildTrials(xDim)

nConditions = 1;

if xDim.explore == 0

    %%% Trial
    x(1, 2, :) = [1 0]; %%% Fixation
    x(1, 3, :) = [0 1]; %%% Target 1
    x(1, 4, :) = [0 1]; %%% Target 2
    x(1, 5, :) = [0 0]; %%% Target 3

    %%% Bias term
    x(1, 1, :) = [ones(1, size(x, 3))];
    
else
    
    %%% Trial
    x(1, 2, :) = [1 0]; %%% Fixation
    x(1, 3, :) = [0 1]; %%% Target 1
    x(1, 4, :) = [0 0]; %%% Target 2
    x(1, 5, :) = [0 1]; %%% Target 3

    %%% Bias term
    x(1, 1, :) = [ones(1, size(x, 3))];
    
end

xcat_t = [];

trialRepeats = 30;

for repeati = 1 : trialRepeats
    trialOrder = randperm(nConditions);
    
    for conditioni = 1 : nConditions
        
        condi = trialOrder(conditioni);
                
        xcat_t = [xcat_t squeeze(x(condi, :, :))];
                
    end
end

T = size(xcat_t, 2);

%%% Zero pad for choices and rewards
xcat_t = [xcat_t; zeros(xDim.nChoices, T); zeros(1, T)];

X = zeros(xDim.nAgents, size(xcat_t, 1), size(xcat_t, 2));

for agenti = 1 : xDim.nAgents
    
    X(agenti, :, :) = xcat_t;
    
end