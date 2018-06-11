load dvfs_dl;

choice = 1;
yStr = '';

fontSize = 15;
figure(1)

%%data of googlenet
s1 = {'googlenet 1600 MHz', 'googlenet 1700 MHz', 'googlenet 1800 MHz', 'googlenet 1900 MHz', 'googlenet 2000 MHz'};
data = googlenet;
data = sortrows(data, [1, 2]);

if choice == 1
    yStr = 'time/s';
    data = data(:, 3);
end

if choice == 2
    yStr = 'power/W';
    data = data(:, 4);
end

if choice == 3
    yStr = 'energy/J';
    data = data(:, 3) .* data(:, 4);
end

data = reshape(data, 4, 5);
data = data';
bar(data,'barWidth',1);
width=900;
height=400;
left=200;
bottom=100;
set(gcf,'position',[left,bottom,width,height]);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s1, 'fontSize', fontSize);
legend({'3500 MHz','4000 MHz','4500 MHz','5000 MHz'},...
    'Location','North',...
    'Orientation','horizontal', 'fontSize', fontSize);
colormap('Summer');
ylabel(yStr, 'fontSize', fontSize+2);

figure(2)

%%data of vgg
s2 = {'vgg 1600 MHz', 'vgg 1700 MHz', 'vgg 1800 MHz', 'vgg 1900 MHz', 'vgg 2000 MHz'};
data = vgg;
data = sortrows(data, [1, 2]);

if choice == 1
    yStr = 'time/s';
    data = data(:, 3);
end

if choice == 2
    yStr = 'power/W';
    data = data(:, 4);
end

if choice == 3
    yStr = 'energy/J';
    data = data(:, 3) .* data(:, 4);
end

data = reshape(data, 4, 5);
data = data';
bar(data,'barWidth',1);
width=900;
height=400;
left=200;
bottom=100;
set(gcf,'position',[left,bottom,width,height]);
ylim([0 max(max(data))*1.2]);
set(gca,'XTickLabel',s2, 'fontSize', fontSize);
legend({'3500 MHz','4000 MHz','4500 MHz','5000 MHz'},...
    'Location','North',...
    'Orientation','horizontal', 'fontSize', fontSize);
colormap('Summer');
ylabel(yStr, 'fontSize', fontSize+2);

