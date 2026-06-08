import json
from pathlib import Path
p=Path('notebooks')
nb_files=sorted(p.glob('*.ipynb'))
changed=[]
for f in nb_files:
    data=json.loads(f.read_text(encoding='utf-8'))
    md=data.setdefault('metadata',{})
    ks=md.get('kernelspec',{})
    name=ks.get('name')
    if name!='predicao-venv':
        md['kernelspec']={'name':'predicao-venv','display_name':'predicao-venv','language':'python'}
        f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        changed.append(f.name)
print('updated', changed)
