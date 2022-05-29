import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from PIL import Image
import numpy as np
from wordcloud import WordCloud
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
import gensim


class DataAna:
    def __init__(self):
        self.path = "/Users/yunhan/Documents/research_project/Academic_conferences" \
                         "/Academic_conferences_among_pandemic/"
        self.data_path = self.path + "Data/"
        self.excel_name = 'Conference_Data.xlsx'
        self.cloud_FC_context = 'BAFC context.txt'
        self.cloud_OC_context = 'BAOC context.txt'
        self.violin_data_sheet_name = 'Violin data'
        self.joint_grid_data_sheet_name = 'JointGrid data'
        self.boxplot_data_sheet_name = 'Boxplot data'
        # self.bar_data_sheet_name = 'Bar data'
        self.violins_data = pd.read_excel(self.data_path + self.excel_name, sheet_name=self.violin_data_sheet_name)
        self.join_grid_data = pd.read_excel(self.data_path + self.excel_name,
                                            sheet_name=self.joint_grid_data_sheet_name)
        self.boxplot_data = pd.read_excel(self.data_path + self.excel_name, sheet_name=self.boxplot_data_sheet_name)
        # self.bar_data = pd.read_excel(self.data_path + self.excel_name, sheet_name=self.bar_data_sheet_name)

    def draw_reason_for_t_test(self):
        file_list_name = ['SN_violin', 'P_violin', 'PD_violin', 'PW_violin', 'CE_violin',
                          # 'FC_other_feature', 'OC_other_feature',
                          # 'P_re', 'FCP_re', 'CE_age_1', 'CE_age_2', 'OCCE_age', 'PD_re', 'PD_e', 'OCPD_age',
                          # 'OCSN_age_1', 'OCSN_age_2', 'OCPD_s', 'FCPD_re',
                          # 'OCSN_age_violin', 'FCPD_education_violin', 'FCPD_research_experience_violin',
                          # 'PD_education_violin',
                          ]
        x_list_name = ['SN Questions', 'P Questions', 'PD Questions', 'PW Questions', 'CE Questions',
                       # 'FC features', 'OC features',
                       # 'P-RE Pair', 'FCP-RE Pair', 'CE-A Pair 1', 'CE-A Pair 2', 'OCCE-A Pair', 'PD-RE Pair',
                       # 'PD-E Pair', 'OCPD-A Pair', 'OCSN-A Pair 1', 'OCSN-A Pair 2', 'OCPD-S Pair', 'FCPD-RE Pair'
                       # 'OC-SN Pair', 'FC-PD Pair', 'Research Experience', 'Education level',
                       ]
        y_list_name = ['SN Score', 'P Score', 'PD Score', 'PW Score', 'CE Score',
                       # 'FC feature scores', 'OC feature scores',
                       # 'P-RE Score', 'FCP Score', 'CE Score 1', 'CE Score 2', 'OCCE Score', 'PD-RE Score', 'PD-E Score',
                       # 'OCPD-A Score', 'OCSN-Age Score 1', 'OCSN-Age Score 2', 'OCPD-S Score', 'FCPD-RE Score'
                       # 'OC-SN Score', 'FC-PD Score', 'FC-PD-RE Score', 'PD-E Score',
                       ]
        hue_list_name = ['SN ConferenceForm', 'P ConferenceForm', 'PD ConferenceForm',
                         'PW ConferenceForm', 'CE ConferenceForm',
                         # None, None,
                         # '3-5 Years(P)', '3-5 Years (FCP)', 'Older than 44', '35-44(CE)', '35-44(OCCE)',
                         # '1-2 Years (PD-RE)', 'Bachelor(PD)', '18-24(OCPD)', '18-24(OCSN)', '35-44(OCSN)',
                         # 'Liberal Arts', '1-2 Years(FCPD-RE)',
                         # 'Age 18-24', 'Bachelor', None, None,
                         ]
        split = [True, True, True, True, True,
                 # False, False,
                 # True, True, True, True, True, True, True, True, True, True, True, True,
                 # True, True, False, False,
                 ]
        for i in range(len(file_list_name)):
            self.violins(x=x_list_name[i], y=y_list_name[i], hue=hue_list_name[i], fig_name=file_list_name[i]+'.pdf',
                         split=split[i])
            plt.cla()
            print('The {0} th fig has finished!'.format(i + 1))

    def violins(self, x, y, hue, fig_name, split=False):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        sns.violinplot(x=x, y=y, hue=hue,
                       data=self.violins_data, palette="pastel", split=split, saturation=0.75, ax=ax)
        if split is True:
            handler, label = ax.get_legend_handles_labels()
            if hue[-14:] == 'ConferenceForm':
                ax.legend(handler, ["FC", "OC"], loc='upper left', bbox_to_anchor=(1, 1))
            else:
                ax.legend(handler, [], frameon=False)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(x, fontdict={'size': 18})
        plt.ylabel(y, fontdict={'size': 18})
        plt.savefig(self.path + 'Code/Output/' + fig_name, bbox_inches='tight')

    def draw_box_line(self):
        box_x_name = ['Age group', 'Age group',
                      'RE group', 'RE group',
                      'Age group',
                      'Age group',
                      'RE group', 'RE group',
                      'E group',
                      'S group']
        box_y_name = ['CE score', 'OCCE score',
                      'P score', 'FCP score',
                      'OCSN score',
                      'OCPD score',
                      'FCPD score', 'PD score',
                      'PD score',
                      'OCPD-S score']
        line_y_name = ['CE-line', 'OCCE line',
                       'P-line', 'FCP line',
                       'OCSN line',
                       'OCPD line_age',
                       'FCPD line_re', 'PD-line_re',
                       'PD-line_e',
                       'OCPD line_s'
                       ]
        file_name_list = ['box_line_CE', 'box_line_OCCE',
                          'box_line_P', 'box_line_FCP',
                          'box_line_OCSN',
                          'box_line_OCPD_Age',
                          'box_line_FCPD_Re', 'box_line_PD_Re',
                          'box_line_PD_E',
                          'box_line_OCPD_S']
        for i in range(len(file_name_list)):
            self.box_line(box_x=box_x_name[i], box_y=box_y_name[i], line_y=line_y_name[i],
                          file_name=file_name_list[i])
            plt.cla()
            print('The {0} th fig has finished!'.format(i + 1))

    def box_line(self, box_x, box_y, line_y, file_name):
        sns.set_theme(style="ticks", palette="pastel")
        fig, ax = plt.subplots(figsize=(16, 4))
        sns.boxplot(x=box_x, y=box_y, data=self.boxplot_data, width=0.2, palette="Blues")
        if box_x == 'Age group':
            line_x_name = 'Age-line'
        elif box_x == 'RE group':
            line_x_name = 'RE-line'
        elif box_x == 'E group':
            line_x_name = 'E-line'
        else:
            line_x_name = 'S-line'
        sns.lineplot(x=line_x_name, y=line_y, data=self.boxplot_data, ax=ax, color='darkorange',
                     linewidth=3.5)  # marker='s'
        if box_x == 'Age group':
            plt.xticks([0, 1, 2, 3], ['18-24', '25-34', '35-44', 'older than 44'])
        elif box_x == 'RE group':
            plt.xticks([0, 1, 2, 3], ['1-2 years', '3-5 years', '6-10 years', 'more than 10 years'])
        elif box_x == 'E group':
            plt.xticks([0, 1, 2], ['Bachelor', 'Master', 'Doctor'])
        else:
            plt.xticks([0, 1, 2], ['Liberal Arts', 'Science', 'Engineering'])
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        ax.set_xlabel(box_x, fontsize=44)
        ax.set_ylabel(box_y, fontsize=44)
        plt.ylim(bottom=0.7, top=7.3)
        plt.grid(axis='y')
        sns.despine()
        plt.savefig(self.path + 'Code/Output/' + file_name + '.pdf', bbox_inches='tight', fliersize=0.5, linewidth=0.1)

    def draw_emotion_context(self):
        file_name_list = ['Changes_content', 'Conference_feature']
        x_list_name = ['Sentiment of change', 'Sentiment of feature']
        y_list_name = ['Aspect of change', 'Aspect of feature']
        for i in range(len(file_name_list)):
            self.draw_joint_grid(x=x_list_name[i], y=y_list_name[i], fig_name=file_name_list[i])
            plt.cla()
            print('The {0} th fig has finished!'.format(i + 1))

    def draw_joint_grid(self, x, y, fig_name):
        sns.set(style="ticks")
        grid_data = self.join_grid_data[[x, y]].dropna(axis=0, how='all')
        g = sns.JointGrid(data=grid_data, x=x, y=y, marginal_ticks=True)
        # Add the joint and marginal histogram plots
        g.plot_joint(sns.histplot, cmap="light:#003366", pmax=.8)
        g.plot_marginals(sns.histplot, color="#003366")
        plt.savefig(self.path + 'Code/Output/' + fig_name + '.pdf', bbox_inches='tight')

    def draw_improvement_compare(self):
        file_name = 'Improvement_compare'
        x = self.bar_data['Possibility of being replaced']
        y1 = self.bar_data['Count1']
        y2 = self.bar_data['Count2']
        y1_label = 'Current'
        y2_label = 'After improvement'
        x_label = self.bar_data['x_label']
        self.plot_bar(x, y1, y2, file_name, y1_label, y2_label, x_label)
        print('Finished!')

    def plot_bar(self, x, y1, y2, file_name, y1_label, y2_label, x_label):
        plt.figure(figsize=(8, 4))
        sns.barplot(x=x, y=y1, color='#FF9933', alpha=1, label=y1_label)
        sns.barplot(x=x, y=y2, color='#0066CC', alpha=0.7, label=y2_label)
        # 网格
        plt.grid(axis='y')
        plt.xticks(list(range(len(x))), x_label)
        # 图例
        plt.legend()
        plt.savefig(self.path + 'Code/Output/' + file_name+'.pdf', bbox_inches='tight')

    def draw_word_fre(self):
        self.draw_word_cloud(self.cloud_FC_context, 'Benefit of FC.png')
        plt.cla()
        self.draw_word_cloud(self.cloud_OC_context, 'Benefit of OC.png')
        plt.cla()
        print('I have finished drawing word could!')

    def draw_word_cloud(self, data_source_name, file_name):
        d = path.dirname(__file__)
        context = open(self.data_path + data_source_name, 'r').read()
        cloud_figure = np.array(Image.open(path.join(d, "cloud2.png")))
        wc = WordCloud(background_color="white", mask=cloud_figure)
        wc.generate(context)
        plt.imshow(wc, interpolation="bilinear")
        wc.to_file(file_name)

    def get_key_list(self, context_file_name):
        key_list = []
        context_list = []
        stopwords = ['able', 'can', 'very', 'are', 'between', 'it', 'that', 'as', 'after', 'to', 'cannot', 'etc',
                     'first', 'be', 'is', 'on', 'you', 'such', 'some', 'same', 'me', 'from', 'around', 'which',
                     'secondly', 'with', 'a', 'for', 'about', 'there', 'being', 'other', 'have', 'then', 'your',
                     'the', 'in', 'and', 'of', 'something', 'at', 'all', 'any', 'beijing', 'also', 'through',
                     'both', 'they', 'or', 'conference', 'example', 's', 'people', 'conference', 'conferences']
        with open(self.data_path + context_file_name, 'r') as f:
            for line in f:
                pro_context = line.replace('_', ' ').replace(',', '').replace('…', '')\
                    .replace(';', '').replace('!', '').replace('.', '').replace('’', ' ')\
                    .replace('\"', '').lower().strip()
                line_word_list = pro_context.split(' ')
                filtered_words = [word for word in line_word_list if word not in stopwords]
                context_list.append(filtered_words)
                for word in filtered_words:
                    if word != '':
                        key_list.append(word)
        key_set = list(set(key_list))
        return key_set, context_list

    def build_matirx(self, context_file_name):
        keys, contexts = self.get_key_list(context_file_name)
        edge = len(keys) + 1
        matrix = [['' for j in range(edge)] for i in range(edge)]
        return matrix, keys, contexts

    def init_matrix(self, context_file_name):
        matrix, set_key_list, contexts = self.build_matirx(context_file_name)
        matrix[0][1:] = np.array(set_key_list)
        matrix = list(map(list, zip(*matrix)))
        matrix[0][1:] = np.array(set_key_list)
        return matrix, contexts, set_key_list

    def count_matrix(self, matrix_file_name, excel_file_name, context_file_name):
        '''计算各个关键词共现次数'''
        matrix, formated_data, keys = self.init_matrix(context_file_name)
        for row in range(1, len(matrix)):
            # 遍历矩阵第一行，跳过下标为0的元素
            for col in range(1, len(matrix)):
                # 遍历矩阵第一列，跳过下标为0的元素
                # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
                if matrix[0][row] == matrix[col][0]:
                    # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
                    matrix[col][row] = str(0)
                else:
                    counter = 0
                    # 初始化计数器
                    for ech in formated_data:
                        # 遍历格式化后的原始数据，让取出的行关键词和取出的列关键词进行组合，
                        # 再放到每条原始数据中查询
                        if matrix[0][row] in ech and matrix[col][0] in ech:
                            counter += 1
                        else:
                            continue
                    matrix[col][row] = str(counter)

        np.savetxt(self.path + 'Code/Output/' + matrix_file_name, matrix, fmt=('%s,' * len(matrix))[:-1])
        data_df = pd.DataFrame(matrix)
        writer = pd.ExcelWriter(self.path + 'Code/Output/' + excel_file_name)
        data_df.to_excel(writer, 'page_1')
        writer.save()

    def word_matrix(self):
        matrix_file_name = ['matrix_fc.txt', 'matrix_oc.txt']
        excel_file_name = ['excel_fc.xlsx', 'excel_oc.xlsx']
        context_file_name_list = [self.cloud_FC_context, self.cloud_OC_context]
        for i in range(len(matrix_file_name)):
            self.count_matrix(matrix_file_name[i], excel_file_name[i], context_file_name_list[i])
            print('The {0} th matrix has finished!'.format(i + 1))

    def testLDA(self):
        # 处理数据，英文文本数据去停用词，计数存放字典，没有做词干提取，因为要与gephi得到的结果做对比
        key_set, content_list = self.get_key_list(self.cloud_FC_context)
        common_dictionary = Dictionary(content_list)
        common_corpus = [common_dictionary.doc2bow(text) for text in content_list]
        lda = LdaModel(corpus=common_corpus, num_topics=3, id2word=common_dictionary, random_state=1)
        for topic in lda.print_topics(num_words=15):  # 这里是打印LDA分类的结果
            print(topic[1])


if __name__ == '__main__':
    data_ana = DataAna()
    # data_ana.draw_box_line()
    data_ana.draw_reason_for_t_test()
    # data_ana.draw_emotion_context()
    # data_ana.draw_improvement_compare()
    # data_ana.draw_word_fre()
    # data_ana.word_matrix()
    # data_ana.testLDA()
