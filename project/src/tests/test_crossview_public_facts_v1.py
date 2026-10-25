"""Offline AST/Drive literal parsing boundaries, with synthetic source text."""
import json
import unittest
from project.src.tools.verify_crossview_public_facts_v1 import literal_configuration,drive_entries


class PublicFactTests(unittest.TestCase):
    def test_both_assignment_forms_and_actual_assignment_lines(self):
        text="import model\nclass C:\n model: str = 'example'\n img_size: int = 384\n checkpoint_start = 'one.pth'\n"
        result=literal_configuration(text)
        self.assertEqual(result['model'],{'value':'example','line':3})
        self.assertEqual(result['checkpoint_start'],{'value':'one.pth','line':5})

    def test_expressions_and_calls_are_not_executed_or_guessed(self):
        text="checkpoint_start = __import__('os').system('must not run')\nimg_size = 192*2\n"
        self.assertEqual(literal_configuration(text),{})

    def test_drive_literal_decoding_not_regex_filename_guess(self):
        tree=[[['id-123',None,'pretrained.zip','application/zip',17]],None]
        payload=json.dumps(tree).replace('"',r'\x22')
        text="window['_DRIVE_ivd'] = '"+payload+"'; x22pretrained.zip"
        parsed=drive_entries(text)
        self.assertEqual([x['name'] for x in parsed],['pretrained.zip'])
        self.assertEqual(parsed[0]['numeric_metadata_uninterpreted'],{'4':17})

    def test_absent_or_unavailable_listing_never_fabricates_weights(self):
        self.assertEqual(drive_entries('<html>sign in or unavailable</html>'),[])


if __name__=='__main__':unittest.main()
