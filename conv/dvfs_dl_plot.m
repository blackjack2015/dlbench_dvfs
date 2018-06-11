load dvfs_conv;

choice = 7;
yStr = '';
if choice == 6
    yStr = 'power/W';
end

if choice == 7
    yStr = 'GFLOPS';
end

if choice == 8
    yStr = 'GFLOPS/W';
end
           
dataC = cell2mat(dvfs_conv(:, choice));
conv_fft = dataC(1:25);
conv_sgemm = dataC(26:50);
conv_wino = dataC(51:75);

figure(1)

%%data of conv_fft
s1 = {'conv\_fft 600 MHz', 'conv\_fft 800 MHz', 'conv\_fft 1000 MHz', 'conv\_fft 1126 MHz', 'conv\_fft 1300 MHz'};
data = conv_fft;

data = reshape(data, 5, 5);
data = data';
bar(data,'barWidth',1);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s1);
legend({'2100 MHz','2500 MHz','3000 MHz','3500 MHz','3800 MHz'},...
    'Location','Best',...
    'Orientation','horizontal');
colormap('Summer');
ylabel(yStr);

figure(2)

%%data of conv_sgemm
s2 = {'conv\_sgemm 600 MHz', 'conv\_sgemm 800 MHz', 'conv\_sgemm 1000 MHz', 'conv\_sgemm 1126 MHz', 'conv\_sgemm 1300 MHz'};
data = conv_sgemm;

data = reshape(data, 5, 5);
data = data';
bar(data,'barWidth',1);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s2);
legend({'2100 MHz','2500 MHz','3000 MHz','3500 MHz','3800 MHz'},...
    'Location','Best',...
    'Orientation','horizontal');
colormap('Summer');
ylabel(yStr);

figure(3)

%%data of conv_winograd
% s3 = {'conv\_winograd 600 MHz', 'conv\_winograd 800 MHz', 'conv\_winograd 1000 MHz', 'conv\_winograd 1126 MHz', 'conv\_winograd 1300 MHz'};
s3 = {'600 MHz', '800 MHz', '1000 MHz', '1126 MHz', '1300 MHz'};
data = conv_wino;

data = reshape(data, 5, 5);
data = data';
bar(data,'barWidth',1);
width=850;
height=350;
left=200;
bottom=100;
set(gcf,'position',[left,bottom,width,height]);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s3, 'fontSize', 14);
legend({'2100 MHz','2500 MHz','3000 MHz','3500 MHz','3800 MHz'},...
    'Location','north',...
    'Orientation','horizontal',...
    'fontSize', 12);
colormap('Summer');
ylabel(yStr, 'fontSize', 14);