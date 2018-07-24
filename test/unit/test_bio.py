"""Tests for bio module"""
import unittest
from sloika import bio


class BioTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('* Testing Bio')
        self.base_seq = 'ATCGATCGATCGATCG'
        self.base_seq_rc = 'CGATCGATCGATCGAT'
        self.kmers1 = ['ATCGATCGAT', 'TCGATCGATC', 'CGATCGATCG', 'GATCGATCGA',
                       'ATCGATCGAT', 'TCGATCGATC', 'CGATCGATCG']
        self.kmers2 = ['ATCGATCGAT', 'ATCGATCGAT', 'CGATCGATCG', 'GATCGATCGA',
                       'GATCGATCGA', 'TCGATCGATC', 'CGATCGATCG']
        self.moves2 = [0, 2, 1, 0, 2, 1]

    def test_all_kmers_returns_correct_kmers(self):
        result = bio.all_kmers(2, alphabet='ATC')
        self.assertEqual(result, ['AA', 'AT', 'AC',
                                  'TA', 'TT', 'TC',
                                  'CA', 'CT', 'CC'])

    def test_kmer_mapping_can_work_with_byte_strings(self):
        self.assertTrue(b'A' in bio.kmer_mapping(1, alphabet=b'A'))

    def test_kmer_mapping_can_work_with_unicode(self):
        self.assertTrue(u'A' in bio.kmer_mapping(1, alphabet=u'A'))

    def test_all_kmers_can_work_with_byte_strings(self):
        self.assertTrue(b'A' in bio.all_kmers(1, alphabet=b'A'))

    def test_all_kmers_can_work_with_unicode(self):
        self.assertTrue(u'A' in bio.all_kmers(1, alphabet=u'A'))

    def test_all_kmers_reverse_map(self):
        kmers = bio.all_kmers(2, alphabet='ATC')
        kmap = bio.kmer_mapping(2, alphabet='ATC')
        for i, k in enumerate(kmers):
            self.assertTrue(kmap[k] == i)

    def test_all_multimers_returns_correct_kmers(self):
        result = bio.all_multimers(2, alphabet='ATC')
        self.assertEqual(result, ['', 'A', 'T', 'C',
                                  'AA', 'AT', 'AC',
                                  'TA', 'TT', 'TC',
                                  'CA', 'CT', 'CC'])

    def test_all_multimers_reverse_map(self):
        kmers = bio.all_multimers(2, alphabet='ATC')
        kmap = bio.multimer_mapping(2, alphabet='ATC')
        for i, k in enumerate(kmers):
            self.assertTrue(kmap[k] == i)

    def test_complement_returns_correct_in_common_cases(self):
        self.assertEqual(bio.complement('A'), 'T')
        self.assertEqual(bio.complement('T'), 'A')
        self.assertEqual(bio.complement('C'), 'G')
        self.assertEqual(bio.complement('G'), 'C')
        self.assertEqual(bio.complement('a'), 't')
        self.assertEqual(bio.complement('t'), 'a')
        self.assertEqual(bio.complement('c'), 'g')
        self.assertEqual(bio.complement('g'), 'c')

    def test_complement_returns_input_for_nonsense_and_prints_warning(self):
        with self.assertRaises(KeyError):
            bio.complement('AC')
        with self.assertRaises(KeyError):
            bio.complement('Q')
        with self.assertRaises(KeyError):
            bio.complement('>')

    def test_reverse_complement(self):
        self.assertEqual(bio.reverse_complement(self.base_seq), self.base_seq_rc)

    def test_reverse_complement_kmer_sequence(self):
        self.assertEqual(bio.reverse_complement_kmers(['ATC', 'TCG']), ['CGA', 'GAT'])

    def test_seq_to_kmers_returns_correct(self):
        self.assertEqual(bio.seq_to_kmers(self.base_seq, 10), self.kmers1)

    def test_de_bruijn(self):
        #  Note: this test is specific to the order produced by Untangled de Bruijn
        de_bruijn_seq = ''.join([str(y) for y in bio.de_bruijn(4, 2)])
        self.assertEqual(de_bruijn_seq, '0010203112132233')

    def test_de_bruijn_allkmers(self):
        alpha = 4
        dblen = 2
        debruijn_seq = ''.join([str(y) for y in bio.de_bruijn(alpha, dblen, pad=True)])
        kmers = bio.seq_to_kmers(debruijn_seq, dblen)
        self.assertEqual(len(kmers), alpha ** dblen)

    def test_de_bruijn_noduplicates(self):
        alpha = 4
        dblen = 2
        debruijn_seq = ''.join([str(y) for y in bio.de_bruijn(alpha, dblen, pad=True)])
        all_kmers = bio.seq_to_kmers(debruijn_seq, dblen)
        self.assertTrue(len(all_kmers) == len(set(all_kmers)))

    def test_max_overlap_simple(self):
        moves = bio.max_overlap(self.kmers1)
        self.assertTrue(all([x == 1 for x in moves]))

    def test_max_overlap_complex(self):
        moves = bio.max_overlap(self.kmers2)
        self.assertTrue(all(map(lambda x, y: x == y, moves, self.moves2)))

    def test_moves_compatible(self):
        compat = bio.moves_compatible(self.kmers2, self.moves2)
        self.assertTrue(all(compat))

    def test_moves_compatible_shifted(self):
        # Also valid due to construction of list of kmers
        moves = [x + 4 for x in self.moves2]
        compat = bio.moves_compatible(self.kmers2, moves)
        self.assertTrue(all(compat))

    def test_moves_incompatible(self):
        moves = [x + 1 for x in self.moves2]
        compat = bio.moves_compatible(self.kmers2, moves)
        self.assertTrue(all([not x for x in compat]))

    def test_moves_single_incompatible(self):
        moves = list(self.moves2)
        moves[3] += 1
        compat = bio.moves_compatible(self.kmers2, moves)
        self.assertFalse(compat[3])

    def test_moves_homopolymer(self):
        KLEN = 10
        homo = ['A' * KLEN] * 2
        for i in range(KLEN + 1):
            self.assertTrue(bio.moves_compatible(homo, [i])[0])

    def test_reduce_kmers(self):
        seq = bio.reduce_kmers(self.kmers2, self.moves2)
        self.assertTrue(seq == self.base_seq)

    def test_reduce_kmers_shifted(self):
        seq = bio.reduce_kmers(self.kmers2, [x + 4 for x in self.moves2])
        seq2 = 'ATCG' * 10
        self.assertTrue(seq == seq2)

    def test_k2s_is_max_reduction(self):
        seq1 = bio.reduce_kmers(self.kmers2, self.moves2)
        seq2 = bio.kmers_to_sequence(self.kmers2)
        self.assertTrue(seq1 == seq2)

    def test_reduce_kmer_regression(self):
        moves = [0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0]
        kmers = ['CGACGA', 'CGACGA', 'CGACGA', 'CGACGA', 'CGACGA', 'CGACGA',
                 'CGACGA', 'CGACGA', 'ACGATG', 'CGATGG', 'CGATGG', 'GATGGA',
                 'GATGGA', 'ATGGAC', 'ATGGAC', 'GGACCA', 'GGACCA', 'GACCAC',
                 'GACCAC']
        seq1 = bio.reduce_kmers(kmers, moves)
        seq2 = 'CGACGATGGACCAC'
        self.assertTrue(seq1 == seq2)

    def test_kmer_transitions_forward_only(self):
        transitions = bio.kmer_transitions(['ATGC', 'TCGA'], proposed_max_move=1, forward_only=True)
        expected = {'ATGC': [(0, 'ATGC'), (1, 'TGCA'), (1, 'TGCC'), (1, 'TGCG'),
                             (1, 'TGCT')],
                    'TCGA': [(0, 'TCGA'), (1, 'CGAA'), (1, 'CGAC'), (1, 'CGAG'),
                             (1, 'CGAT')]}

        self.assertDictEqual(expected, transitions)

    def test_kmer_transitions(self):
        kmers = ['ATG']
        transitions1 = bio.kmer_transitions(kmers, proposed_max_move=2, forward_only=False)
        expected = {'ATG': [(0, 'ATG'), (1, 'TGA'), (1, 'TGC'), (1, 'TGG'),
                            (1, 'TGT'), (-1, 'AAT'), (-1, 'CAT'), (-1, 'GAT'),
                            (-1, 'TAT'), (2, 'GAA'), (2, 'GAC'), (2, 'GAG'),
                            (2, 'GAT'), (2, 'GCA'), (2, 'GCC'), (2, 'GCG'),
                            (2, 'GCT'), (2, 'GGA'), (2, 'GGC'), (2, 'GGG'),
                            (2, 'GGT'), (2, 'GTA'), (2, 'GTC'), (2, 'GTG'),
                            (2, 'GTT'), (-2, 'AAA'), (-2, 'ACA'), (-2, 'AGA'),
                            (-2, 'ATA'), (-2, 'CAA'), (-2, 'CCA'), (-2, 'CGA'),
                            (-2, 'CTA'), (-2, 'GAA'), (-2, 'GCA'), (-2, 'GGA'),
                            (-2, 'GTA'), (-2, 'TAA'), (-2, 'TCA'), (-2, 'TGA'),
                            (-2, 'TTA')]}

        self.assertDictEqual(expected, transitions1)
        # check we get the same if proposed_max_move is too larger for the kmer length
        transitions2 = bio.kmer_transitions(kmers, proposed_max_move=3, forward_only=False)
        self.assertDictEqual(expected, transitions2)


if __name__ == '__main__':
    unittest.main()
