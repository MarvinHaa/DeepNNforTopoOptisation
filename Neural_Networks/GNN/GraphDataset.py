import dgl
from dgl.data import DGLDataset
import os

from torch.utils.data import DataLoader


def ConvGraphLoad(dir='data/'):
    graph_list = os.listdir(dir)
    graph_list_tuple = [tuple(graph_name.split('_', 2)) for graph_name in graph_list]
    graph_list_tuple.sort(key=lambda tup: int(tup[1]))
    graph_list_tuple.sort(key=lambda tup: int(tup[0]))
    print('Graph List sorted:', graph_list_tuple)
    glist = []
    labellist = []

    for i in range(len(graph_list_tuple) - 2):
        graph_step_name = graph_list_tuple[i][0] + '_' + graph_list_tuple[i][1] + '_' + graph_list_tuple[i][2]
        graph_next_step_name = graph_list_tuple[i + 1][0] + '_' + graph_list_tuple[i + 1][1] + '_' + graph_list_tuple[i + 1][2]

        g, _ = dgl.load_graphs(dir + graph_step_name)
        g = g[0]

        g_next, _ = dgl.load_graphs(dir + graph_next_step_name)
        g_next = g_next[0]

        if g.number_of_nodes() == g_next.number_of_nodes() and int(graph_list_tuple[i][1]) <= 491 and int(graph_list_tuple[i][1]) > 250:
            glist.append(g)
            labellist.append(g_next.ndata['value'])


    return glist, labellist


def ConvLabeldGraphLoad(dir='../u_label_plus_10/'):
    graph_list = os.listdir(dir)  # kick DS_Store files -> find cleaner way

    glist = []

    for graph_name in graph_list:

        g, _ = dgl.load_graphs(dir + graph_name)
        g = g[0]

        glist.append(g)

    return glist


class GraphDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self,
                 url=None,
                 raw_dir='data/',
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(GraphDataset, self).__init__(name='',
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)



    def process(self):
        mat_path = self.raw_path

        self.graphs = ConvLabeldGraphLoad(mat_path)


    def __getitem__(self, idx):

        return self.graphs[idx], self.label[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)


    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 1


if __name__ == '__main__':


    dataset = GraphDataset()


    def _collate_fn(batch):
        g, label = batch[0]

        # g = dgl.batch(batch, edata=None)
        # print(g)
        # label = g.ndata['labels']
        return g, label


    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)

    i = 0
    for g, labels in dataloader:
        i += 1
        g.ndata['labels'] = labels[:, 0]
        dgl.save_graphs('data_labeld_late/' + str(i) + '.bin', g)
