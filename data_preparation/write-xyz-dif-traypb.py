import numpy as np 
import os, sys , re

file_in = sys.argv[1]

x0=5.1692309943199817e+00 
x1=9.4865769005676327e+01
y0=9.6170786247926259e+00 
y1=9.0327921375198656e+01
z0=2.0000000000000000e-03 
z1=3.3991999999999997e+01

head = '436\nLattice=\"%10.8f 0.0 0.0 0.0 %10.8f 0.0 0.0 0.0 %10.8f\"  Properties=species:S:1:pos:R:3:tags:I:1:forces:R:3:Z:I:1 energy=0.00000000 pbc=\"T T T\"  \n'%(x1-x0,y1-y0,z1-z0) #:Z:I:1
text =''
time =0

count_line = 0
counting=0
print("Working directory is:", os.getcwd())
with open(file_in,'r') as f:
    for line in f:
        count_line +=1
        max_words=len(line.split())
        items = re.split('\s+',line)

        if items[0] == '9.6170786247926259e+00':
            time +=1
            text = text+head
            print('time ',time)
            name2 = './head.dat'
            cg_dna = np.loadtxt(name2, usecols=(0, 1, 2, 3, 4, 5, 6))
            count = 0

            for j in range(len(cg_dna)):
                if (cg_dna[j,2] == 1):
                    x = float(cg_dna[j,3])
                    y = float(cg_dna[j,4])
                    z = float(cg_dna[j,5])
                    dx = x - x0
                    dy = y - y0
                    dz = z - z0
                    text += 'C  %12.5f  %12.5f  %12.5f %5s 0 %5s %12.8f  %12.8f  %12.8f 6 \n' % (dx, dy, dz,'     ', '     ', 0.0, 0.0, 0.0)  # efx,efy,efz)
                if (cg_dna[j,2] == 2):
                    x = float(cg_dna[j,3])
                    y = float(cg_dna[j,4])
                    z = float(cg_dna[j,5])
                    dx = x - x0
                    dy = y - y0
                    dz = z - z0
                    text += 'N  %12.5f  %12.5f  %12.5f %5s 0 %5s %12.8f  %12.8f  %12.8f 7 \n' % (dx, dy, dz,'     ', '     ', 0.0, 0.0, 0.0)  # efx,efy,efz)
                if (cg_dna[j,2] == 3):
                    x = float(cg_dna[j,3])
                    y = float(cg_dna[j,4])
                    z = float(cg_dna[j,5])
                    dx = x - x0
                    dy = y - y0
                    dz = z - z0
                    text += 'O  %12.5f  %12.5f  %12.5f %5s 0 %5s %12.8f  %12.8f  %12.8f 8 \n' % (dx, dy, dz,'     ', '     ', 0.0, 0.0, 0.0)  # efx,efy,efz)
                if (cg_dna[j,2] == 4):
                    x = float(cg_dna[j, 3])
                    y = float(cg_dna[j, 4])
                    z = float(cg_dna[j, 5])
                    dx = x - x0
                    dy = y - y0
                    dz = z - z0
                    text += 'P  %12.5f  %12.5f  %12.5f %5s 0 %5s %12.8f  %12.8f  %12.8f 15 \n' % (dx, dy, dz,'     ','     ', 0.0, 0.0, 0.0)  # efx,efy,efz)

        if ((items[0] != 'ITEM:') and len(items) > 8):

            if items[2] == '25':
                x = float(items[3])
                y = float(items[4])
                z = float(items[5])

                dx = x - x0
                dy = y - y0
                dz = z - z0


                fx = float(items[6])-(float(items[9]))
                fy = float(items[7])-(float(items[10]))
                fz = float(items[8])-(float(items[11]))

                text +='Na %12.5f  %12.5f  %12.5f %5s 1 %5s  %12.8f  %12.8f  %12.8f 11 \n'%(dx,dy,dz,'     ', '     ',fx,fy,fz) #efx,efy,efz)



            if items[2] == '24':
                x = float(items[3])
                y = float(items[4])
                z = float(items[5])

                dx = x-x0
                dy = y-y0
                dz = z-z0
                fx = float(items[6])-(float(items[9]))
                fy = float(items[7])-(float(items[10]))
                fz = float(items[8])-(float(items[11]))

                text +='Cl %12.5f  %12.5f  %12.5f %5s 1 %5s %12.8f  %12.8f  %12.8f 17 \n'%(dx,dy,dz,'     ', '     ',fx,fy,fz) #efx,efy,efz)



with open("config_for_train.xyz", "w", encoding="utf-8") as fd:
    fd.write(text)

fd.close()
