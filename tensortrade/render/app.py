import dash
import dash_core_components as dcc
import dash_html_components as html

#custom css stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#dict for specifying colors 
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


#webpage structure 
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='DeepTrade',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(
            children='Dash: A web interface for trading.', 
            style={
            'textAlign': 'center',
            'color': colors['text']
    }),

    dcc.Graph(
        id='results',
        figure={
            'data': [ #callback function for data 
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'typ 'bar', 'name': u'Montréal'},e':
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
])


if __name__ == '__main__':
    #debug must be true for hot reloading 
    app.run_server(debug=True)