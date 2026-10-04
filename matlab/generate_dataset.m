clear; clc; close all;
rng(42);

VERBOSE_DIAG = false;

carrier_freq = 3.5e9;
wavelength   = physconst('LightSpeed') / carrier_freq;

SCS          = 15e3;
NFFT         = 1024;
sample_rate  = NFFT * SCS;
N_RB         = 52;
N_sc_used    = N_RB * 12;
symbolsPerSlot = 14;

cpLengths = 72 * ones(1, symbolsPerSlot);
cpLengths([1, 8]) = 80;
assert(sum(NFFT + cpLengths) == sample_rate/1000, 'CP schedule mismatch.');

half = N_sc_used / 2;
usedIdx = [2:(1+half), (NFFT-half+1):NFFT];

pilotSymbolIdx   = 6;
combSpacing      = 4;
pilotSubcarriers = 1:combSpacing:N_sc_used;
pilotValue       = 1 + 0j;

fprintf('\n--- OFDM Numerology ---\n');
fprintf('SCS = %.0f kHz, NFFT = %d, fs = %.2f MHz\n', SCS/1e3, NFFT, sample_rate/1e6);
fprintf('Occupied BW = %.2f MHz (complex baseband Nyquist = %.2f MHz)\n', ...
    N_sc_used*SCS/1e6, sample_rate/1e6);
fprintf('CSI-RS-like pilot: comb-%d on symbol %d -> %d pilot REs\n\n', ...
    combSpacing, pilotSymbolIdx, numel(pilotSubcarriers));

sysParams = struct('carrier_freq', carrier_freq, 'wavelength', wavelength, ...
    'SCS', SCS, 'NFFT', NFFT, 'sample_rate', sample_rate, 'N_sc_used', N_sc_used, ...
    'symbolsPerSlot', symbolsPerSlot, 'cpLengths', cpLengths, 'usedIdx', usedIdx, ...
    'pilotSymbolIdx', pilotSymbolIdx, 'pilotSubcarriers', pilotSubcarriers, ...
    'pilotValue', pilotValue, 'combSpacing', combSpacing);

antenna_configs = [4,2; 8,4; 16,8; 4,4; 8,2];

common_snr_range = [-15, 25];

scenarios(1).name='Indoor Office';         scenarios(1).cdl_profile='CDL-A';
scenarios(1).delay_spread_range=[10e-9,50e-9];   scenarios(1).snr_range=common_snr_range;
scenarios(1).probability=0.25;             scenarios(1).velocity_range=[0,3];

scenarios(2).name='Outdoor Urban LOS';     scenarios(2).cdl_profile='CDL-B';
scenarios(2).delay_spread_range=[50e-9,150e-9];  scenarios(2).snr_range=common_snr_range;
scenarios(2).probability=0.20;             scenarios(2).velocity_range=[1,10];

scenarios(3).name='Outdoor Urban NLOS';    scenarios(3).cdl_profile='CDL-C';
scenarios(3).delay_spread_range=[100e-9,300e-9]; scenarios(3).snr_range=common_snr_range;
scenarios(3).probability=0.35;             scenarios(3).velocity_range=[5,20];

scenarios(4).name='Outdoor Canyon';        scenarios(4).cdl_profile='CDL-D';
scenarios(4).delay_spread_range=[200e-9,400e-9]; scenarios(4).snr_range=common_snr_range;
scenarios(4).probability=0.20;             scenarios(4).velocity_range=[10,33.3];

impProb = struct('doppler',1.00,'phase_noise',0.80,'pa_nonlinear',0.60, ...
                 'iq_imbalance',0.50,'colored_noise',0.40,'interference',0.30);

impParams = struct( ...
    'pa_alpha_range',[0.05,0.20], 'iq_gain_db_range',[0.3,1.5], ...
    'iq_phase_deg_range',[1,5], 'phase_noise_dbc',-80, ...
    'colored_rho_range',[0.3,0.8], 'sir_db_range',[5,20], ...
    'impulsive_prob',0.10, 'impulsive_frac',0.02);

modulation_schemes = {'QPSK','16QAM','64QAM'};

num_samples_main = 100000;
fprintf('=== MAIN dataset (%d samples) ===\n', num_samples_main);
[X_main, meta_main] = generate_ofdm_dataset(num_samples_main, scenarios, antenna_configs, ...
    modulation_schemes, impProb, impParams, sysParams, VERBOSE_DIAG);

feature_names_16 = {'EstChannelGain','K_factor_PDP_dB','K_factor_Moment_dB', ...
    'Est_RMS_DelaySpread_ns','Est_Coherence_BW_MHz','Est_Num_Paths', ...
    'Est_Max_Delay_ns','FreqDomain_GainVariability_dB', ...
    'Rx_Power','Rx_Power_Std','Rx_Phase_Mean', ...
    'Equalized_Signal_Power','Equalized_Signal_Std','Kurtosis', ...
    'Num_Tx_Antennas','Num_Rx_Antennas'};
assert(size(X_main,2)==16,'X must be 16 columns');

main_table = array2table(X_main,'VariableNames',feature_names_16);
main_table.SNR_dB = meta_main.snr_true;
writetable(main_table,'Enhanced_5G_Dataset_100K_v311.csv');

meta_table = table(meta_main.snr_true,meta_main.snr_ls,meta_main.snr_ml, ...
    meta_main.snr_evm,meta_main.snr_dd, ...
    meta_main.cdl_profile,meta_main.mod_scheme,meta_main.doppler_hz, ...
    meta_main.flag_pa,meta_main.flag_iq,meta_main.flag_pn,meta_main.flag_cn,meta_main.flag_intf, ...
    'VariableNames',{'SNR_true','SNR_LS','SNR_ML','SNR_EVM','SNR_DD', ...
    'CDL_Profile','Mod_Scheme','Doppler_Hz','Flag_PA','Flag_IQ','Flag_PN','Flag_CN','Flag_Intf'});
writetable(meta_table,'Enhanced_5G_Dataset_100K_v3_metadata11.csv');

fprintf('\nSaved:\n');
fprintf('  Enhanced_5G_Dataset_100K_v311.csv\n');
fprintf('  Enhanced_5G_Dataset_100K_v3_metadata11.csv\n');

function [X, meta] = generate_ofdm_dataset(numSamples, scenarios, antenna_configs, ...
    modulation_schemes, impProb, impParams, sysParams, verbose)

    X = zeros(numSamples,16);
    snr_true=zeros(numSamples,1); snr_ls=zeros(numSamples,1);
    snr_ml=zeros(numSamples,1);   snr_evm=zeros(numSamples,1);
    snr_dd=zeros(numSamples,1);
    cdl_profile_all=cell(numSamples,1); mod_scheme_all=cell(numSamples,1);
    doppler_hz_all=zeros(numSamples,1);
    flag_pa_all=zeros(numSamples,1); flag_iq_all=zeros(numSamples,1);
    flag_pn_all=zeros(numSamples,1); flag_cn_all=zeros(numSamples,1);
    flag_intf_all=zeros(numSamples,1);

    probs = [scenarios.probability];
    n_per = round(numSamples*probs);
    n_per(end) = numSamples - sum(n_per(1:end-1));

    idx = 1;
    for sIdx = 1:numel(scenarios)
        scn = scenarios(sIdx);
        nThis = n_per(sIdx);
        fprintf('  [%s | %s] %d samples...\n', scn.name, scn.cdl_profile, nThis);
        for s = 1:nThis
            if mod(s,5000)==0, fprintf('    %d / %d\n', s, nThis); end

            cfgIdx = randi(size(antenna_configs,1));
            Nt = antenna_configs(cfgIdx,1); Nr = antenna_configs(cfgIdx,2);

            velocity  = scn.velocity_range(1) + diff(scn.velocity_range)*rand();
            f_doppler = velocity * sysParams.carrier_freq / physconst('LightSpeed');

            cdl = nrCDLChannel;
            cdl.DelayProfile = scn.cdl_profile;
            cdl.DelaySpread  = scn.delay_spread_range(1) + diff(scn.delay_spread_range)*rand();
            cdl.CarrierFrequency = sysParams.carrier_freq;
            cdl.TransmitAntennaArray.Size = [Nt 1 1 1 1];
            cdl.ReceiveAntennaArray.Size  = [Nr 1 1 1 1];
            cdl.SampleRate = sysParams.sample_rate;
            cdl.MaximumDopplerShift = f_doppler;
            cdl.NormalizePathGains = true;
            cdl.NormalizeChannelOutputs = false;

            antenna_spacing = (0.4 + 0.2*rand()) * sysParams.wavelength;
            tx_angle = -60 + 120*rand();
            w_tx = exp(-1j*2*pi*antenna_spacing/sysParams.wavelength*(0:Nt-1)'*sind(tx_angle))/sqrt(Nt);

            target_snr_db = scn.snr_range(1) + diff(scn.snr_range)*rand();
            mod_scheme = modulation_schemes{randi(3)};
            switch mod_scheme
                case 'QPSK',  mod_order = 4;
                case '16QAM', mod_order = 16;
                case '64QAM', mod_order = 64;
            end

            nSc = sysParams.N_sc_used; nSym = sysParams.symbolsPerSlot;
            X_grid = zeros(nSc, nSym);
            nDataRE = nSc*nSym - numel(sysParams.pilotSubcarriers);
            switch mod_scheme
                case 'QPSK'
                    bits = randi([0 1], nDataRE*2, 1);
                    ints = bi2de(reshape(bits,2,[]).','left-msb');
                    dataSyms = pskmod(ints, mod_order, pi/4);
                case '16QAM'
                    bits = randi([0 1], nDataRE*4, 1);
                    ints = bi2de(reshape(bits,4,[]).','left-msb');
                    dataSyms = qammod(ints, mod_order, 'UnitAveragePower', true);
                case '64QAM'
                    bits = randi([0 1], nDataRE*6, 1);
                    ints = bi2de(reshape(bits,6,[]).','left-msb');
                    dataSyms = qammod(ints, mod_order, 'UnitAveragePower', true);
            end
            isPilotRE = false(nSc, nSym);
            isPilotRE(sysParams.pilotSubcarriers, sysParams.pilotSymbolIdx) = true;
            X_grid(~isPilotRE) = dataSyms;
            X_grid(isPilotRE)  = sysParams.pilotValue;

            Xprecoded = reshape(X_grid,[nSc,nSym,1]) .* reshape(w_tx,[1,1,Nt]);
            txWave = ofdm_modulate(Xprecoded, sysParams.usedIdx, sysParams.NFFT, sysParams.cpLengths);

            flag_pa = rand() < impProb.pa_nonlinear;
            if flag_pa
                a_range = impParams.pa_alpha_range;
                alpha_pa = a_range(1) + diff(a_range)*rand();
                txWave = txWave ./ (1 + alpha_pa*abs(txWave).^2);
                txWave = txWave ./ sqrt(mean(abs(txWave(:)).^2) + 1e-12);
            end

            flag_iq = rand() < impProb.iq_imbalance;
            if flag_iq
                g_range = impParams.iq_gain_db_range; ph_range = impParams.iq_phase_deg_range;
                gain_imb_dB   = g_range(1) + diff(g_range)*rand();
                phase_imb_deg = ph_range(1) + diff(ph_range)*rand();
                g = 10^(gain_imb_dB/20); ph = phase_imb_deg*pi/180;
                a_iq = (1 + g*exp(1j*ph))/2; b_iq = (1 - g*exp(1j*ph))/2;
                txWave = a_iq*txWave + b_iq*conj(txWave);
            end

            chInfo = info(cdl);
            filtDelay = chInfo.ChannelFilterDelay;
            txWavePadded = [txWave; zeros(filtDelay+32, size(txWave,2))];
            [rxRaw,~,~] = cdl(txWavePadded);
            rxWave = rxRaw(filtDelay+1 : filtDelay+size(txWave,1), :);

            SNR_lin = 10^(target_snr_db/10);
            sigPower = mean(abs(rxWave(:)).^2);
            sigPowerPerSc = sigPower * (sysParams.NFFT / sysParams.N_sc_used);
            noisePower = Nr * sigPowerPerSc / SNR_lin;

            flag_cn = rand() < impProb.colored_noise;
            noise = complex(zeros(size(rxWave)));
            for r = 1:Nr
                if flag_cn
                    rho_range = impParams.colored_rho_range;
                    rho = rho_range(1) + diff(rho_range)*rand();
                    nWhite = sqrt(noisePower/2)*(randn(size(rxWave,1),1)+1j*randn(size(rxWave,1),1));
                    nCol = zeros(size(rxWave,1),1); nCol(1)=nWhite(1);
                    for k = 2:numel(nCol)
                        nCol(k) = rho*nCol(k-1) + sqrt(1-rho^2)*nWhite(k);
                    end
                    nCol = nCol * sqrt(noisePower/(mean(abs(nCol).^2)+1e-12));
                    noise(:,r) = nCol;
                else
                    noise(:,r) = sqrt(noisePower/2)*(randn(size(rxWave,1),1)+1j*randn(size(rxWave,1),1));
                end
                if rand() < impParams.impulsive_prob
                    nImp = ceil(size(rxWave,1)*impParams.impulsive_frac);
                    impIdx = randperm(size(rxWave,1), nImp);
                    impPwr = noisePower * (5+10*rand());
                    noise(impIdx,r) = noise(impIdx,r) + ...
                        sqrt(impPwr/2)*(randn(nImp,1)+1j*randn(nImp,1));
                end
            end

            flag_intf = rand() < impProb.interference;
            interference = complex(zeros(size(rxWave)));
            if flag_intf
                sir_range = impParams.sir_db_range;
                SIR_dB = sir_range(1) + diff(sir_range)*rand();
                intfPower = sigPower / 10^(SIR_dB/10);
                for r = 1:Nr
                    interference(:,r) = sqrt(intfPower)*(2*randi([0 1],size(rxWave,1),1)-1);
                end
            end

            rxNoisy = rxWave + noise + interference;

            flag_pn = rand() < impProb.phase_noise;
            if flag_pn
                target_slot_drift_deg = 3 * 10^((impParams.phase_noise_dbc + 80)/20);
                target_slot_drift_rad = target_slot_drift_deg * pi/180;
                nSampPN = size(rxNoisy,1);
                sigma_pn = target_slot_drift_rad / sqrt(nSampPN);
                dphi = sigma_pn * randn(nSampPN,1);
                phiPN = cumsum(dphi);
                rxNoisy = rxNoisy .* exp(1j*phiPN);
            end

            Y = ofdm_demodulate(rxNoisy, sysParams.usedIdx, sysParams.NFFT, sysParams.cpLengths);

            nPilot = numel(sysParams.pilotSubcarriers);
            hEstPilotsPerAnt = reshape(Y(sysParams.pilotSubcarriers, sysParams.pilotSymbolIdx, :), ...
                                       nPilot, Nr) / sysParams.pilotValue;

            noiseBinStart = ceil(nPilot/4);
            noiseBinEnd   = ceil(3*nPilot/4);
            noiseBinIdx   = noiseBinStart:noiseBinEnd;

            hEstDenoisedPerAnt = complex(zeros(nPilot, Nr));
            noiseVarPerAnt     = zeros(1, Nr);
            pdpPerAnt          = zeros(nPilot, Nr);

            for r = 1:Nr
                pdpTime_r = ifft(hEstPilotsPerAnt(:,r));
                pdp_r     = abs(pdpTime_r).^2;
                pdpPerAnt(:,r) = pdp_r;
                noiseVarPerAnt(r) = nPilot * mean(pdp_r(noiseBinIdx));
                noiseFloor_r = mean(pdp_r(noiseBinIdx));
                validBins_r  = pdp_r > 5 * noiseFloor_r;
                if ~any(validBins_r)
                    validBins_r = false(size(pdp_r)); validBins_r(1) = true;
                end
                pdpTimeDenoised_r = pdpTime_r;
                pdpTimeDenoised_r(~validBins_r) = 0;
                hEstDenoisedPerAnt(:,r) = fft(pdpTimeDenoised_r);
            end
            noiseVarEst = mean(noiseVarPerAnt);

            pdp = mean(pdpPerAnt, 2);
            pdpPeak = max(pdp);
            thresh = pdpPeak * 10^(-10/10);
            deltaTau_ns = 1e9 / (nPilot * sysParams.combSpacing * sysParams.SCS);
            delayAxis_ns = ((0:nPilot-1) * deltaTau_ns).';
            validBins = pdp > thresh;
            if any(validBins)
                tau = delayAxis_ns(validBins); p = pdp(validBins);
                meanDelay_ns = sum(tau.*p)/sum(p);
                rmsDS_ns = sqrt(sum(((tau-meanDelay_ns).^2).*p)/sum(p));
                maxDelayEst_ns = max(tau);
                numPathsEst = sum(validBins);
                Kpdp_dB = 10*log10(pdpPeak / max(sum(pdp)-pdpPeak, 1e-12));
            else
                rmsDS_ns = 1; maxDelayEst_ns = deltaTau_ns; numPathsEst = 1; Kpdp_dB = 30;
            end
            Kpdp_dB = max(-10, min(30, Kpdp_dB));
            cohBW_MHz = 1 / (2*pi*max(rmsDS_ns,1e-3)*1e-9) / 1e6;

            combinedMagPilots = sqrt(sum(abs(hEstDenoisedPerAnt).^2, 2));
            m2 = mean(combinedMagPilots.^2); m4 = mean(combinedMagPilots.^4);
            innerTerm = 2*m2^2 - m4;
            if innerTerm > 0
                Klin = sqrt(innerTerm) / max(m2 - sqrt(innerTerm), 1e-12);
                Kmoment_dB = 10*log10(max(Klin, 1e-3));
            else
                Kmoment_dB = 30;
            end
            Kmoment_dB = max(-10, min(30, Kmoment_dB));
            freqGainVar_dB = std(20*log10(combinedMagPilots + 1e-12));

            allPos = 1:nSc;
            hEstFullPerAnt = complex(zeros(nSc, Nr));
            for r = 1:Nr
                hReal_r = interp1(sysParams.pilotSubcarriers, real(hEstDenoisedPerAnt(:,r)), allPos, 'linear', 'extrap');
                hImag_r = interp1(sysParams.pilotSubcarriers, imag(hEstDenoisedPerAnt(:,r)), allPos, 'linear', 'extrap');
                hEstFullPerAnt(:,r) = (hReal_r + 1j*hImag_r).';
            end

            gComb = sum(abs(hEstFullPerAnt).^2, 2);
            yComb = zeros(nSc, nSym);
            for r = 1:Nr
                yComb = yComb + conj(hEstFullPerAnt(:,r)) .* Y(:,:,r);
            end
            eqGrid = yComb ./ max(gComb, 1e-12);

            chPowerEst = mean(gComb);

            dataMask = ~isPilotRE;
            rxData = yComb(dataMask);
            eqData = eqGrid(dataMask);
            txDataRef = dataSyms;

            snr_est_ls = clip_db(10*log10(chPowerEst / max(noiseVarEst,1e-12)), target_snr_db);
            sigma2_ml = mean(mean(abs(hEstPilotsPerAnt - hEstDenoisedPerAnt).^2, 1));
            snr_est_ml = clip_db(10*log10(chPowerEst / max(sigma2_ml,1e-12)), target_snr_db);

            errEVM = eqData - txDataRef;
            evm2 = mean(abs(errEVM).^2) / (mean(abs(txDataRef).^2)+1e-12);
            snr_est_evm = clip_db(10*log10(1/max(evm2,1e-12)), 50);

            switch mod_scheme
                case 'QPSK'
                    ints_dd = pskdemod(eqData,4,pi/4); xhat_dd = pskmod(ints_dd,4,pi/4);
                case '16QAM'
                    ints_dd = qamdemod(eqData,16,'UnitAveragePower',true); xhat_dd = qammod(ints_dd,16,'UnitAveragePower',true);
                case '64QAM'
                    ints_dd = qamdemod(eqData,64,'UnitAveragePower',true); xhat_dd = qammod(ints_dd,64,'UnitAveragePower',true);
            end
            eDD = eqData - xhat_dd;
            errPowDD = mean(abs(eDD).^2); sigPowDD = mean(abs(xhat_dd).^2);
            snr_est_dd = clip_db(10*log10(sigPowDD/max(errPowDD,1e-12)), 50);

            rxKurt = mean(abs(rxData).^4) / (mean(abs(rxData).^2)^2 + 1e-30);
            feat = [ chPowerEst, Kpdp_dB, Kmoment_dB, rmsDS_ns, cohBW_MHz, numPathsEst, ...
                     maxDelayEst_ns, freqGainVar_dB, ...
                     mean(abs(rxData).^2), std(abs(rxData).^2), mean(angle(rxData)), ...
                     mean(abs(eqData).^2), std(abs(eqData)), rxKurt, ...
                     Nt, Nr ];

            X(idx,:) = feat;

            if verbose
                raw_ls = 10*log10(chPowerEst / max(noiseVarEst,1e-12));
                fprintf(['n=%3d Nr=%2d tgt=%7.2f rawLS=%7.2f ' ...
                         'chP=%9.3e nV=%9.3e noiseP=%9.3e sigP=%8.3e ' ...
                         'nV/noiseP=%7.3f chP/(Nr*sigP)=%7.3f\n'], ...
                    idx, Nr, target_snr_db, raw_ls, ...
                    chPowerEst, noiseVarEst, noisePower, sigPower, ...
                    noiseVarEst/max(noisePower,1e-12), ...
                    chPowerEst/max(Nr*sigPower,1e-12));
            end

            snr_true(idx) = target_snr_db;
            snr_ls(idx) = snr_est_ls; snr_ml(idx) = snr_est_ml;
            snr_evm(idx) = snr_est_evm; snr_dd(idx) = snr_est_dd;
            cdl_profile_all{idx} = scn.cdl_profile;
            mod_scheme_all{idx}  = mod_scheme;
            doppler_hz_all(idx)  = f_doppler;
            flag_pa_all(idx)=double(flag_pa); flag_iq_all(idx)=double(flag_iq);
            flag_pn_all(idx)=double(flag_pn); flag_cn_all(idx)=double(flag_cn);
            flag_intf_all(idx)=double(flag_intf);

            idx = idx + 1;
        end
    end

    meta = struct('snr_true',snr_true,'snr_ls',snr_ls,'snr_ml',snr_ml, ...
        'snr_evm',snr_evm,'snr_dd',snr_dd, ...
        'cdl_profile',{cdl_profile_all},'mod_scheme',{mod_scheme_all}, ...
        'doppler_hz',doppler_hz_all,'flag_pa',flag_pa_all,'flag_iq',flag_iq_all, ...
        'flag_pn',flag_pn_all,'flag_cn',flag_cn_all,'flag_intf',flag_intf_all);
end

function txWave = ofdm_modulate(Xg, usedIdx, NFFT, cpLengths)
    [~, nSym, nAnt] = size(Xg);
    symLen = NFFT + cpLengths;
    txWave = complex(zeros(sum(symLen), nAnt));
    for a = 1:nAnt
        ptr = 1;
        for l = 1:nSym
            grid = zeros(NFFT,1);
            grid(usedIdx) = Xg(:,l,a);
            td = ifft(grid) * sqrt(NFFT);
            cp = td(end-cpLengths(l)+1:end);
            txWave(ptr:ptr+symLen(l)-1, a) = [cp; td];
            ptr = ptr + symLen(l);
        end
    end
end

function Y = ofdm_demodulate(rxWave, usedIdx, NFFT, cpLengths)
    nAnt = size(rxWave,2);
    nSym = numel(cpLengths);
    symLen = NFFT + cpLengths;
    Y = complex(zeros(numel(usedIdx), nSym, nAnt));
    for a = 1:nAnt
        ptr = 1;
        for l = 1:nSym
            seg = rxWave(ptr:ptr+symLen(l)-1, a);
            td = seg(cpLengths(l)+1:end);
            Xf = fft(td) / sqrt(NFFT);
            Y(:,l,a) = Xf(usedIdx);
            ptr = ptr + symLen(l);
        end
    end
end

function out = clip_db(val, fallback)
    if ~isfinite(val) || ~isreal(val), val = fallback; end
    out = max(-30, min(50, real(val)));
end
