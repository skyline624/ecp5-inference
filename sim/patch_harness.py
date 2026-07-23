import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
old = '#include <libgen.h>  // basename'
new = ('// basename (libgen.h POSIX, absent sous MSVC -> impl inline)\n'
       '#include <cstring>\n'
       'static const char* vb_basename(const char* p){\n'
       '    const char* b = p;\n'
       '    for (const char* q = p; *q; ++q) if (*q == \'/\' || *q == \'\\\\\') b = q + 1;\n'
       '    return b;\n'
       '}')
assert old in s, "libgen include not found in " + p
s = s.replace(old, new)
s = s.replace('basename(argv[0])', 'vb_basename(argv[0])')
open(p, 'w', encoding='utf-8').write(s)
print('patched ' + p)