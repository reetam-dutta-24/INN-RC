function INNRC_App
% INNRC_APP  GUI for the INN-RC project (App Designer components).
%   Load a grayscale image, type a secret message, embed it with
%   PEE / baseline float INN / proposed IC-INN, extract it again and check
%   that the cover comes back bit for bit. An attack menu shows what
%   happens to the message on a lossy channel.
%   Run from the matlab/ folder:  >> INNRC_App

root = fileparts(fileparts(mfilename('fullpath')));
app.prop = icinn_load(fullfile(root, 'models', 'icinn.mat'));
app.base = finn_load(fullfile(root, 'models', 'finn_s4.mat'));
app.cover = []; app.stego = []; app.msgBits = []; app.root = root;

fig = uifigure('Name', 'INN-RC : Lossless Image Steganography', 'Position', [80 80 1250 720]);
g = uigridlayout(fig, [3 4]);
g.RowHeight = {'1x', '1x', 210};
g.ColumnWidth = {'1x', '1x', '1x', 330};

ax(1) = uiaxes(g); ax(1).Layout.Row = 1; ax(1).Layout.Column = 1; title(ax(1), 'Cover');
ax(2) = uiaxes(g); ax(2).Layout.Row = 1; ax(2).Layout.Column = 2; title(ax(2), 'Stego');
ax(3) = uiaxes(g); ax(3).Layout.Row = 1; ax(3).Layout.Column = 3; title(ax(3), '|Stego - Cover| x 10');
ax(4) = uiaxes(g); ax(4).Layout.Row = 2; ax(4).Layout.Column = 1; title(ax(4), 'Received (after channel)');
ax(5) = uiaxes(g); ax(5).Layout.Row = 2; ax(5).Layout.Column = 2; title(ax(5), 'Recovered cover');
ax(6) = uiaxes(g); ax(6).Layout.Row = 2; ax(6).Layout.Column = 3; title(ax(6), 'Recovered - Cover');
for k = 1:6, axis(ax(k), 'off'); end

% ---- control panel ----
p = uipanel(g, 'Title', 'Controls'); p.Layout.Row = [1 2]; p.Layout.Column = 4;
pg = uigridlayout(p, [12 2]);
pg.RowHeight = repmat({28}, 1, 12);
uibutton(pg, 'Text', 'Load image', 'ButtonPushedFcn', @(~,~) onLoad());
uibutton(pg, 'Text', 'Sample image', 'ButtonPushedFcn', @(~,~) onSample());
uilabel(pg, 'Text', 'Method');
ddMethod = uidropdown(pg, 'Items', {'Proposed (IC-INN)', 'Baseline (float INN)', 'Classical PEE'});
uilabel(pg, 'Text', 'Payload');
ddPay = uidropdown(pg, 'Items', {'256 bits (s=8)', '1024 bits (s=4)', '4096 bits (s=2)'}, 'Value', '1024 bits (s=4)');
uilabel(pg, 'Text', 'Secret key');
efKey = uieditfield(pg, 'numeric', 'Value', 20261127);
uilabel(pg, 'Text', 'Message');
efMsg = uieditfield(pg, 'text', 'Value', 'UCT702 secret message');
uibutton(pg, 'Text', 'Embed', 'ButtonPushedFcn', @(~,~) onEmbed());
uibutton(pg, 'Text', 'Save stego', 'ButtonPushedFcn', @(~,~) onSave());
uilabel(pg, 'Text', 'Channel');
ddAtt = uidropdown(pg, 'Items', {'Lossless', 'Gaussian noise (std 2)', 'Salt & pepper (1%)', 'JPEG Q90'});
uibutton(pg, 'Text', 'Extract', 'ButtonPushedFcn', @(~,~) onExtract());
uibutton(pg, 'Text', 'Clear', 'ButtonPushedFcn', @(~,~) onClear());
lblStatus = uilabel(pg, 'Text', 'Load an image to start');
lblStatus.Layout.Row = 9; lblStatus.Layout.Column = [1 2];

txt = uitextarea(g, 'Editable', 'off', 'FontName', 'Consolas');
txt.Layout.Row = 3; txt.Layout.Column = [1 4];

    function onLoad()
        [f, d] = uigetfile({'*.png;*.bmp;*.tif;*.pgm;*.jpg', 'Images'});
        if isequal(f, 0), return; end
        setCover(imread(fullfile(d, f)));
    end

    function onSample()
        setCover(imread(fullfile(app.root, 'data', 'test_kodak', 'kodim23.png')));
    end

    function setCover(I)
        if size(I, 3) == 3, I = rgb2gray(I); end
        I = im2uint8(I);
        sz = 16 * floor(size(I) / 16);           % must be a multiple of 2*s
        app.cover = I(1:sz(1), 1:sz(2));
        app.stego = [];
        imshow(app.cover, 'Parent', ax(1));
        for k = 2:6, cla(ax(k)); end
        lblStatus.Text = sprintf('Image %d x %d loaded', sz(1), sz(2));
    end

    function s = payloadS()
        v = ddPay.Value;
        s = str2double(v(strfind(v, 's=') + 2));
    end

    function onEmbed()
        if isempty(app.cover), lblStatus.Text = 'Load an image first'; return; end
        s = payloadS();
        if startsWith(ddMethod.Value, 'Baseline') && s ~= app.base.s
            s = app.base.s; ddPay.Value = '1024 bits (s=4)';
        end
        nb = numel(app.cover) / 4 / s^2;
        app.msgBits = text2bits(efMsg.Value, nb);
        lblStatus.Text = 'Embedding...'; drawnow;
        try
            tic;
            switch ddMethod.Value
                case 'Proposed (IC-INN)'
                    [S, info] = inncr_embed(app.cover, app.msgBits, app.prop, s, efKey.Value);
                case 'Baseline (float INN)'
                    [S, info] = baseline_embed(app.cover, app.msgBits, app.base, efKey.Value);
                otherwise
                    [S, info] = pee_embed(app.cover, app.msgBits); info.aux_bits = 0;
            end
            t = toc;
        catch err
            lblStatus.Text = 'Embedding failed (see log)';
            txt.Value = {['Embedding failed: ' err.message]};
            return;
        end
        app.stego = S; app.s = s;
        imshow(S, 'Parent', ax(2));
        imshow(uint8(min(255, 10 * abs(double(S) - double(app.cover)))), 'Parent', ax(3));
        txt.Value = {sprintf('Method: %s   payload: %d bits (%.4f bpp)   time: %.2f s', ddMethod.Value, nb, nb/numel(S), t), ...
                     sprintf('PSNR = %.2f dB   SSIM = %.4f   side information = %d bits', ...
                             metric_psnr(S, app.cover), metric_ssim(S, app.cover), info.aux_bits)};
        lblStatus.Text = 'Embedded';
    end

    function onExtract()
        if isempty(app.stego), lblStatus.Text = 'Embed first'; return; end
        switch ddAtt.Value
            case 'Lossless',               A = app.stego;
            case 'Gaussian noise (std 2)', A = apply_attack(app.stego, 'gauss', 2);
            case 'Salt & pepper (1%)',     A = apply_attack(app.stego, 'sp', 0.01);
            otherwise,                     A = apply_attack(app.stego, 'jpeg', 90);
        end
        imshow(A, 'Parent', ax(4));
        lossless = strcmp(ddAtt.Value, 'Lossless');
        I2 = []; key = efKey.Value; s = app.s;
        try
            switch ddMethod.Value
                case 'Proposed (IC-INN)'
                    if lossless
                        [bits, I2] = inncr_extract(A, app.prop, s, key);
                    else
                        L = icinn_decode(app.prop, A, s);
                        bits = xor(L(:) > 0, logistic_keystream(numel(L), key));
                    end
                case 'Baseline (float INN)'
                    if lossless
                        [bits, I2] = baseline_extract(A, app.base, key);
                    else
                        [~, mr] = finn_reveal(app.base, double(A) / 255);
                        L = block_mean(mr, s);
                        bits = xor(L(:) > 0, logistic_keystream(numel(L), key));
                    end
                otherwise
                    [bits, I2] = pee_extract(A);
                    bits = bits(1:numel(app.msgBits));
            end
        catch err
            txt.Value = [txt.Value; {['Extraction failed: ' err.message]}];
            lblStatus.Text = 'Extraction failed'; return;
        end
        [ber, nc] = metric_ber_nc(app.msgBits, bits);
        lines = {sprintf('Channel: %s   BER = %.4f   NC = %.4f', ddAtt.Value, ber, nc), ...
                 ['Recovered message: ' bits2text(bits)]};
        if ~isempty(I2)
            imshow(I2, 'Parent', ax(5));
            imshow(uint8(128 + double(I2) - double(app.cover)), 'Parent', ax(6));
            lines{end+1} = sprintf('Cover recovered exactly: %s   (PSNR recovered vs cover = %g dB)', ...
                                   mat2str(isequal(I2, app.cover)), metric_psnr(I2, app.cover));
        else
            cla(ax(5)); cla(ax(6));
            lines{end+1} = 'Lossy channel: only the message is read, the cover cannot be restored.';
        end
        txt.Value = [txt.Value; lines'];
        lblStatus.Text = 'Extracted';
    end

    function onSave()
        if isempty(app.stego), return; end
        [f, d] = uiputfile('stego.png');
        if isequal(f, 0), return; end
        imwrite(app.stego, fullfile(d, f));
    end

    function onClear()
        app.stego = []; txt.Value = {''};
        for k = 2:6, cla(ax(k)); end
    end
end

function bits = text2bits(str, nb)
% 16-bit length + ASCII bits, the rest padded with zeros
b = dec2bin(double(str), 8)' - '0';
bits = [dec2bin(numel(str), 16) - '0', b(:)'];
if numel(bits) > nb, bits = bits(1:nb); end
bits = logical([bits, zeros(1, nb - numel(bits))])';
end

function str = bits2text(bits)
bits = double(bits(:)');
n = bits(1:16) * 2.^(15:-1:0)';
n = min(n, floor((numel(bits) - 16) / 8));
c = reshape(bits(17:16 + 8*n), 8, n)';
str = char(c * 2.^(7:-1:0)')';
str(str < 32 | str > 126) = '?';
end
