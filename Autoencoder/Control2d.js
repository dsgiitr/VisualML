function Controls2D(obj) {

  // parse input arguments
  obj = obj || {};
  this.min = obj.min || 0;
  this.max = obj.max || 100;
  this.onDrag = obj.onDrag || function(a) {console.log('* sampling', a)};
  this.container = obj.container


  // state
  this.initial = null; // initial coords of toggle
  this.down = null; // object if we've mousedowned, else null

  // create the ui
  this.render = function() {
    this.box = document.createElement('div');
    this.box.id = 'ui-box';
    this.toggle = document.createElement('div');
    this.toggle.id = 'ui-toggle';
    this.box.appendChild(this.toggle);
    this.container.appendChild(this.box);
  }

  // style the ui
  this.style = function() {
    var style = document.createElement('style');
    style.textContent = '#ui-box {' +
      '  position: fixed;' +
      '  width: 100px;' +
      '  height: 100px;' +
      '  background: #eee;' +
      '  border: 1px solid #7b7b7b;' +
      '  box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.3);' +
      '  top: 40px;' +
      '  right: 40px;' +
      '}' +
      '#ui-toggle {' +
      '  width: 16px;' +
      '  height: 16px;' +
      '  background: #fff;' +
      '  border: 1px solid #909090;' +
      '  border-radius: 100%;' +
      '  position: relative;' +
      '  display: inline-block;' +
      '  box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.3);' +
      '}';
    document.head.appendChild(style);
  };

  // add main function called on toggle drag
  this.moveToggle = function(e) {
    if (!this.initial) return;
    var x = e.clientX - this.initial.x,
        y = e.clientY - this.initial.y,
        boxW = this.box.clientWidth,
        toggleW = this.toggle.clientWidth + 2; // 2px border
    // keep toggle in box
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > boxW - toggleW) x = boxW - toggleW;
    if (y > boxW - toggleW) y = boxW - toggleW;
    this.toggle.style.left = x + 'px';
    this.toggle.style.top = y + 'px';
    // pass scaled x, y coords to obj.onDrag callback
    this.onDrag({
      x: this.scale(x),
      y: this.scale(y),
    });
  }

  // add event listeners to controls
  this.addListeners = function() {

    this.toggle.addEventListener('mousedown', function(e) {
      if (!this.initial) this.initial = {x: e.clientX, y: e.clientY};
      this.down = {x: e.clientX, y: e.clientY};
    }.bind(this))

    document.addEventListener('mousemove', function(e) {
      if (this.down) {
        this.moveToggle(e);
        e.preventDefault();
        e.stopPropagation();
      }
    }.bind(this))

    document.addEventListener('mouseup', function(e) {
      this.moveToggle(e);
      this.down = null;
    }.bind(this))
  }

  // scale sampled value `v` from 0:82 to 0:1
  this.scale = function(v) {
    return v/82;
  }

  this.style();
  this.render();
  this.addListeners();
}
