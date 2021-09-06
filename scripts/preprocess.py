import pandas as pd

n = 0
with open('../data/BindingDB_All.tsv') as binddb:
    with open('../data/BindingDB_clean.tsv','w+') as cleanbinddb:
        for line in binddb:
            if len(line.split("\t")) > 193:
                n += 1
            else:
                cleanbinddb.write(line)
        cleanbinddb.close()
    binddb.close()
print(n)

bindind_df = pd.read_csv('../data/BindingDB_clean.tsv',sep= '\t',index_col=None)

singlepair_df = bindind_df[bindind_df['Number of Protein Chains in Target (>1 implies a multichain complex)']==1]

sele_df = singlepair_df[['BindingDB Reactant_set_id',
               'BindingDB Ligand Name',
               'Ki (nM)',
               'IC50 (nM)',
               'Kd (nM)',
               'EC50 (nM)',
               'kon (M-1-s-1)',
               'koff (s-1)',
               'pH',
               'Temp (C)',
               'Ligand HET ID in PDB',
               'PDB ID(s) for Ligand-Target Complex',
               'PDB ID(s) of Target Chain',
               'KEGG ID of Ligand',
               'ZINC ID of Ligand',
                'PubChem CID',
               'Ligand SMILES',
               'BindingDB Target Chain  Sequence']]
               
sele_df['Protein ID'] = sele_df.apply(lambda x: str(hash(x['BindingDB Target Chain  Sequence'])).replace("-","_"),axis=1)
clean_singlepair_df = sele_df[sele_df['Protein ID'] != "0"]

fasta_dcit = dict(zip(clean_singlepair_df['Protein ID'],clean_singlepair_df['BindingDB Target Chain  Sequence']))

def filterprotein(prot_seq):
    for AA in prot_seq:
        if AA in 'QWERTYIPASDFGHKLCVNM':
            continue
        else:
            return 0


with open('../data/single_pair.fasta','w+') as fasta:
    for key in fasta_dcit:
        try:
            seq = ">"+str(key)+"\n"+fasta_dcit[key].upper()+"\n"
            fasta.write(seq)
        except TypeError:
            print(key,fasta_dcit[key])
    fasta.close()

def fasta2dic(fastafilename): #read a fasta file into a dict
    fasta_dict = {}
    with open(fastafilename) as fastafile:
        for line in fastafile:
            if line[0] == ">":
                head = line.strip()
                fasta_dict[head] = ''
            else:
                fasta_dict[head] += line.strip()
        fastafile.close()
    return fasta_dict

fa_dict = fasta2dic('../data/single_pair.fasta')
new_dict = {}
for key in fa_dict:
    if filterprotein(fa_dict[key]) != 0:
        new_dict[key] = fa_dict[key]

def dump_fasta(fa_dict,outfile_name):
    with open(outfile_name,'w+') as outfile_fasta:
        for key in fa_dict:
            try:
                if key[0] == ">":
                    seq = str(key)+"\n"+fa_dict[key].upper()+"\n"
                else:
                    seq = ">"+str(key)+"\n"+fa_dict[key].upper()+"\n"
                outfile_fasta.write(seq)
            except TypeError:
                print(key,fa_dict[key])
        outfile_fasta.close()

dump_fasta(new_dict,'../data/single_pair_clean.fasta')

clean_singlepair_df.to_csv('../data/single_pair_data.tsv',sep='\t')
clean_singlepair_df[['Ligand SMILES','Protein ID']].to_csv('../data/LigPro_pairwise.tsv',sep='\t')

