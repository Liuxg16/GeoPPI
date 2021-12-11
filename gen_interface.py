import glob, os, random
import sys
from pymol import cmd
import InterfaceResidues

pdbobject=  sys.argv[1]
namepdb = os.path.basename(pdbobject)
name = namepdb.split('.')[0]

interface_info=  sys.argv[2]
chainsAB = interface_info.split('_')
chainsAB = chainsAB[0]+chainsAB[1]

workdir=  sys.argv[3]
cmd.load(pdbobject)
interfaces= []

for i in range(len(chainsAB)):
	for j in range(i+1,len(chainsAB)):
		cha,chb=chainsAB[i],chainsAB[j]
		if cha==chb:continue
		cmd.do('interfaceResidue {}, chain {}, chain {}'.format(name, cha, chb))
		mapp = {'chA':cha,'chB':chb}
		ffile = open('temp/temp.txt','r')
		for line in ffile.readlines():
			linee = line.strip().split('_')
			resid = linee[0]
			chainn = mapp[linee[1]]
			inter='{}_{}_{}_{}'.format(cha,chb,chainn,resid)
			if inter not in interfaces:
				interfaces.append(inter)
		os.system('rm temp/temp.txt')
ffile = open('{}/interface.txt'.format(workdir),'w')
for x in interfaces:
	ffile.write(x+'\n')
cmd.save(pdbobject)
cmd.delete('all')


