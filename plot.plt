unset border
unset xtics
unset ytics
set arrow 1 from 1,1 to 1.5,(1+sqrt(3)/2) nohead front lt -1 lw 1
set arrow 2 from 1,1 to 2,1          nohead front lt -1 lw 1
set arrow 3 from 2,1 to 1.5,(1+sqrt(3)/2) nohead front lt -1 lw 1

rgb(r,g,b) = 65536 * int(r) + 256 * int(g) + int(b)

#set label 1 "1" at 0.5,sqrt(3)/2+.05
#set label 2 "2" at 1+.05,0
#set label 3 "3" at -.05,0
plot 'output.csv' using 1:2:3 with points ls 7 lc palette

pause -1