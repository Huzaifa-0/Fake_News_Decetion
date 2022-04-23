import 'dart:convert';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'widgets/widgets.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _textController = TextEditingController();
  final _textFieldFocusNode = FocusNode();
  var _isLoading = false;
  String? _text, _result;
  BorderSide borderSide = BorderSide.none;
  static const _url = 'https://fakenewsdec.herokuapp.com';

  Future<void> _submit() async {
    setState(() {
      _isLoading = true;
    });

    var apiUrl = Uri.parse('$_url/predict');
    _text = _textController.text;

    try {
      final response = await http.post(
        apiUrl,
        body: json.encode({'text': _text}),
        headers: {'Content-Type': 'application/json; charset=UTF-8'},
      );
      final jsonData = json.decode(response.body) as Map<String, dynamic>;
      _result = jsonData['result'] as String;
    } catch (e) {
      debugPrint(e.toString());
      setState(() {
        _isLoading = false;
      });
    }
    debugPrint(_result);
    setState(() {
      _isLoading = false;
      if (_result == null) return;
      if (_result == 'real') {
        borderSide = const BorderSide(color: Colors.green, width: 2.5);
      } else {
        borderSide = const BorderSide(color: Colors.red, width: 2.5);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return SafeArea(
      child: Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            image: DecorationImage(
              image: AssetImage('assets/bg.jpg'),
              fit: BoxFit.cover,
            ),
          ),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 3, sigmaY: 3),
            child: SingleChildScrollView(
              child: SizedBox(
                height: (mediaQuery.size.height - mediaQuery.padding.top),
                width: (mediaQuery.size.width),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    buildTitle(),
                    const Spacer(
                      flex: 2,
                    ),
                    buildTextField(
                        _textController, _textFieldFocusNode, borderSide),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Button(
                          text: 'Submit',
                          onPress: () async {
                            if (_textController.text.isNotEmpty) {
                              FocusScope.of(context).requestFocus(FocusNode());
                              await _submit();
                            }
                          },
                        ),
                        const SizedBox(
                          width: 16,
                        ),
                        Button(
                          text: 'Reset',
                          onPress: () {
                            FocusScope.of(context).requestFocus(FocusNode());
                            _textController.clear();
                            setState(() {
                              borderSide = BorderSide.none;
                              _result = null;
                            });
                          },
                        ),
                      ],
                    ),
                    const SizedBox(
                      height: 16,
                    ),
                    ResultContainer(result: _result, isLoading: _isLoading),
                    const Spacer(
                      flex: 2,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
