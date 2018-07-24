import glob
from nose_parameterized import parameterized
import os
import unittest


from fast5_research import Fast5, iterate_fast5


class IterationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']
        self.readdir = os.path.join(self.dataDir, 'reads')
        self.strand_list = os.path.join(self.dataDir, 'strands.txt')
        self.basenames = [
            'read1.fast5',
            'read2.fast5',
            'read3.fast5',
            'read4.fast5',
            'read5.fast5',
            'read6.fast5',
            'read7.fast5',
            'read8.fast5',
        ]
        self.strands = set([os.path.join(self.readdir, r) for r in self.basenames])

    def test_iterate_returns_all(self):
        fast5_files = set(iterate_fast5(self.readdir, paths=True))
        dir_list = set(glob.glob(os.path.join(self.readdir, '*.fast5')))
        self.assertTrue(fast5_files == dir_list)

    def test_iterate_respects_limits(self):
        _LIMIT = 2
        fast5_files = set(iterate_fast5(self.readdir, paths=True, limit=_LIMIT))
        self.assertTrue(len(fast5_files) == _LIMIT)

    def test_iterate_works_with_strandlist(self):
        fast5_files = set(iterate_fast5(self.readdir, paths=True,
                                              strand_list=self.strand_list))
        self.assertTrue(self.strands == fast5_files)


class GetAnyMappingDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    def test_unknown(self):
        basename = 'read6'
        filename = os.path.join(self.dataDir, 'reads', basename + '.fast5')

        with Fast5(filename) as f5:
            ev, _ = f5.get_any_mapping_data('template')
            self.assertEqual(len(ev), 10750)


class ReaderAttributesTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    def test_filename_short(self):
        basename = 'read6'
        filename = os.path.join(self.dataDir, 'reads', basename + '.fast5')

        with Fast5(filename) as f5:
            sn = f5.filename_short
            self.assertEqual(f5.filename_short, basename)


class GetSectionEventsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    @parameterized.expand([
        [os.path.join('reads', 'read3.fast5'), 'Segment_Linear', 9946],
        [os.path.join('reads', 'read6.fast5'), 'Segment_Linear', 11145],
    ])
    def test(self, relative_file_path, analysis, number_of_events):
        filename = os.path.join(self.dataDir, relative_file_path)

        with Fast5(filename) as f5:
            ev = f5.get_section_events('template', analysis=analysis)
            self.assertEqual(len(ev), number_of_events)


class GetReadTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    @parameterized.expand([
        [os.path.join('reads', 'read3.fast5'), 51129, True],
        [os.path.join('reads', 'read6.fast5'), 55885, True],
        [os.path.join('reads', 'read2.fast5'), 69443, True],
        [os.path.join('reads', 'read1.fast5'), 114400, True],
    ])
    def test(self, relative_file_path, number_of_events, raw):
        filename = os.path.join(self.dataDir, relative_file_path)

        with Fast5(filename) as f5:
            ev = f5.get_read(raw=raw)
            self.assertEqual(len(ev), number_of_events)
