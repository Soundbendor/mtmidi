import util as UM
import mido
import musicnoteconv as mnc
import polars as pl
import chords as CH

# (x,x,5,4): sus?
# ('min, 1,4,5,1) has minor 5

maj_diatonic = {1: ('c4', 'maj'),
                2: ('d4', 'min'),
                3: ('e4', 'min'),
                4: ('f4', 'maj'),
                5: ('g4', 'maj'),
                6: ('a4', 'min'),
                7: ('b4', 'dim')}

# abuse of "diatonic"
# just make g4 major
min_diatonic = {1: ('c4', 'min'),
                2: ('d4', 'dim'),
                3: ('ef4', 'maj'),
                4: ('f4', 'min'),
                5: ('g4', 'maj'),
                6: ('af4', 'maj'),
                7: ('bf4', 'maj')}

# just switch 4 to minor, 6 to flat6 major, 7 to flat7 major
# althuogh there isn't any flat7 to switch
modemix_diatonic = {1: ('c4', 'maj'),
                2: ('d4', 'min'),
                3: ('e4', 'min'),
                4: ('f4', 'min'),
                5: ('g4', 'maj'),
                6: ('af4', 'maj'),
                7: ('b4', 'dim')}

orig_chordprog = [('maj', 1,4,5,1),
        ('maj' 1,4,6,5),
        ('maj', 1,5,6,4),
        ('maj', 1,6,4,5),
        ('maj', 2,5,1,6),
        ('maj', 4,1,5,6),
        ('maj', 5,4,1,5),
        ('maj', 5,6,4,1),
        ('maj', 6,4,1,5),
        ('min', 1,2,5,1),
        ('min', 1,3,4,1),
        ('min', 1,4,5,1),
        ('min', 1,6,7,1),
        ('min', 1,7,1,1),
        ('min', 6,7,6,1),
        ]



