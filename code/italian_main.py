import argparse
from pathlib import Path
import utils as u


def _select_sentences(args):
	train_split = int(args.sentences_number*0.8)
	dev_split = int(args.sentences_number*0.1)
	test_split = int(args.sentences_number*0.1)

	train_filename = args.input_dir.glob("*train.conllu").__next__()
	dev_filename = args.input_dir.glob("*dev.conllu").__next__()
	test_filename = args.input_dir.glob("*test.conllu").__next__()

	p = Path(args.output_dir)
	p.mkdir(parents=True, exist_ok=True)

	u.sample(train_filename, train_split, p.joinpath("train.conllu"), args.seed)
	u.sample(dev_filename, dev_split, p.joinpath("dev.conllu"), args.seed)
	u.sample(test_filename, test_split, p.joinpath("test.conllu"), args.seed)


if __name__ == "__main__":

	parent_parser = argparse.ArgumentParser(add_help=False)

	root_parser = argparse.ArgumentParser(prog='ud+', add_help=True)
	subparsers = root_parser.add_subparsers(title="actions", dest="actions")

	parser_selectsentences = subparsers.add_parser('select-sentences', parents=[parent_parser],
												description='select suitable sentences from ud treebanks',
												help='select suitable sentences from ud treebanks')
	parser_selectsentences.add_argument("-i", "--input-dir", required=True,
								type=Path,
								help="path to input directory containing UD treebank, has to be in UD standard format")
	parser_selectsentences.add_argument("-o", "--output-dir",
								 required=True, type=Path,
								 help="path to output folder")
	parser_selectsentences.add_argument("-n", "--sentences-number", default=1000, type=int,
								 help="number of sentences to be sampled, they will be sampled as per UD guidelines: 80% from train, 10% from dev and 10% from test")
	parser_selectsentences.add_argument("-s", "--seed", default=1243, type=int,
								 help="seed for random generation")
	parser_selectsentences.set_defaults(func=_select_sentences)

	args = root_parser.parse_args()

	if "func" not in args:
		root_parser.print_usage()
		exit()

	args.func(args)