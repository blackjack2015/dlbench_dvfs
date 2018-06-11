
perf = rawdata(:, 1);
power = rawdata(:, 2);
energy = perf .* power;

figure(1)

%%data of conv_fft
s1 = {'600 MHz', '800 MHz', '1000 MHz', '1126 MHz', '1300 MHz'};
data = perf;

data = reshape(data, 5, 5);
data = data';
bar(data,'barWidth',1);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s1);
legend({'2100 MHz','2500 MHz','3000 MHz','3500 MHz','3800 MHz'},...
    'Location','North',...
    'Orientation','horizontal');
colormap('Summer');
ylabel('time/s');

figure(2)

%%data of conv_fft
s1 = {'600 MHz', '800 MHz', '1000 MHz', '1126 MHz', '1300 MHz'};
data = power;

data = reshape(data, 5, 5);
data = data';
bar(data,'barWidth',1);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s1);
legend({'2100 MHz','2500 MHz','3000 MHz','3500 MHz','3800 MHz'},...
    'Location','North',...
    'Orientation','horizontal');
colormap('Summer');
ylabel('power/W');

figure(3)

%%data of conv_fft
s1 = {'600 MHz', '800 MHz', '1000 MHz', '1126 MHz', '1300 MHz'};
data = energy;

data = reshape(data, 5, 5);
data = data';
bar(data,'barWidth',1);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s1);
legend({'2100 MHz','2500 MHz','3000 MHz','3500 MHz','3800 MHz'},...
    'Location','North',...
    'Orientation','horizontal');
colormap('Summer');
ylabel('energy/J');
