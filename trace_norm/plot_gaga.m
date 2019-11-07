times = {time0';time1';time2';time3';time4';time5';time6';time8';time7'};
perfs = {perf0';perf1(:,1)';perf2(:,1)';perf3(:,1)';perf4(:,1)';perf5(:,1)';perf6(:,1)';perf8(:,1)';perf7(:,1)'};
minV = 0;%min(perf7(:,1));
for i=1:9
    %if i==3
    %   continue
    %end
if length(times{i})>1000
    ind = linspace(1,length(times{i}),100);
    ind = ceil(ind);
else
    ind = 1:length(times{i});
end
plot(times{i}(ind),log(perfs{i}(ind)-minV))
hold on;
%plot(time2,log(perf2(:,1)))
%plot(time3,log(perf3(:,1)))
%plot(time4,log(perf4(:,1)))
%plot(time5,log(perf5(:,1)))
%plot(time6,log(perf6(:,1)))
%plot(time7,log(perf7(:,1)))
end
legend('APG','SVRF', 'SGD', 'SVRG', 'STORC', 'SFW', 'SCGS', 'blockFW','PDFW(ours)');
%legend('SVRF', 'SGD', 'STORC', 'SFW', 'SCGS', 'PDFW(ours)');
%axis([0 10 log(min(perf7(:,1)))+1e-6 log(max(perf1(:,1)))])
axis([0 800 log(min(perf7(:,1))) 4])
%figure
%plot(time1,perf1(:,2))
%hold on;
%plot(time2,perf2(:,2))
%plot(time3,perf3(:,2))
%plot(time4,perf4(:,2))
%plot(time5,perf5(:,2))
%plot(time6,perf6(:,2))
%figure

